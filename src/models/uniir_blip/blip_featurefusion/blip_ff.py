from models.uniir_blip.backbone.med import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

from models.uniir_blip.backbone.blip import create_vit, init_tokenizer, load_checkpoint
from models.uniir_blip.backbone.transform.blip_transform import get_blip_transform


class BLIPFeatureFusion(nn.Module):
    def __init__(
        self,
        med_config="backbone/configs/med_config.json",
        image_size=224,
        vit="base",
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        embed_dim=768,
        queue_size=57600,
        momentum=0.995,
        config=None,  # model config
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        ### 构建视觉编码器（ViT，可开启梯度检查点 vit_grad_ckpt 减显存），并取出视觉通道维度 vision_width。
        # •	初始化文本编码器（BertModel 变体）。encoder_width 告知 cross-attention 对接的视觉嵌入维度。
        # •	初始化 tokenizer（用于后续 get_tokenizer 包装）。
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.image_size = image_size
        self.tokenizer = init_tokenizer()

        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=True)

        # create momentum encoders
        ### 创建动量副本（*_m）：结构相同，但权重由主干 EMA 更新，而非梯度更新。
	    ### •	用 self.model_pairs 管理参数成对关系；copy_params() 初始将动量权重拷贝为主模型权重，并冻结 requires_grad=False。
        self.visual_encoder_m, vision_width = create_vit(vit, image_size)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=True)

        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.text_encoder, self.text_encoder_m],
        ]
        self.copy_params()

        # create the queue
        ### 队列与指针	
        # •	注册为 buffer（随模型移动与保存，不参与梯度）：
        # •	query_queue/cand_queue: 存放动量编码器输出的特征（对比库）。
        # •	idx_queue: 存放候选的标识 id（用于软/硬标签计算）。
        # •	new_ptr_queue: 循环队列写指针。
        self.register_buffer("query_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("cand_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1, queue_size), -100))  # [1, queue_size]
        self.register_buffer("new_ptr_queue", torch.zeros(1, dtype=torch.long))

        self.query_queue = nn.functional.normalize(self.query_queue, dim=0)
        self.cand_queue = nn.functional.normalize(self.cand_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.embed_dim = embed_dim
        self.config = config

    def get_img_preprocess_fn(self):
        is_train = self.training
        print(f"Using {'train' if is_train else 'val'} image transform")
        return get_blip_transform(self.image_size, min_scale=0.5, is_train=is_train)

    def get_tokenizer(self):
        def tokenizer_wrapper(txt):
            return self.tokenizer(
                txt,
                padding="max_length",
                truncation=True,
                max_length=self.config.tokenizer_max_length,
                return_tensors="pt",
            )

        return tokenizer_wrapper

    ### 目标：给一批文本+图像做融合编码，输出池化后的句向量/融合向量。
    def encode_multimodal_input(self, txt_dict_batched, image_batched, txt_mask, img_mask, use_momentum=False):
        """encode multimodal input into embeddings

        Args:
            txt_dict_batched
            image_batched (torch.Tensor): images in shape [batch_size, C, H, W]
            txt_mask (torch.Tensor): text attention masks in shape [batch_size, seq_len]
            img_mask (torch.Tensor): image attention masks in shape [batch_size, 1]

        Returns:
            fused_emb (torch.Tensor): fused embeddings in shape [batch_size, embed_dim]
        """
        # We create zero mask for padding image.
        if use_momentum:
            image_embeds_m = self.visual_encoder_m(image_batched)
            image_atts_m = torch.ones(image_embeds_m.size()[:-1], dtype=torch.long).to(image_batched.device)
            fused_emb_m = self.text_encoder_m(
                txt_dict_batched.input_ids,
                attention_mask=txt_dict_batched.attention_mask,
                encoder_hidden_states=image_embeds_m,
                encoder_attention_mask=image_atts_m,
                return_dict=True,
            )
            return fused_emb_m.pooler_output
        else:
            image_embeds = self.visual_encoder(image_batched)
            ### 构造全 1 的 image_atts（形状与视觉 token 对齐）
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_batched.device)
            fused_emb = self.text_encoder(
                txt_dict_batched.input_ids,
                attention_mask=txt_dict_batched.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )  # shape: [batch_size*2 + hard_neg_num*batch_size, embed_dim]
            return fused_emb.pooler_output

    def compute_contrastive_loss(self, batch, alpha):
        ### 从 collator 读入统一打包后的四类张量（文本/图像及其“是否存在”的 mask）。如上所述，后两者当前未显式使用。
        txt_dict_batched = batch["txt_batched"]
        image_batched = batch["image_batched"]
        txt_mask_batched = batch["txt_mask_batched"]
        image_mask_batched = batch["image_mask_batched"]

        ### pc_idx：正例的候选标识（did 的哈希），用于构造软/硬标签。
        pc_idx = torch.tensor(batch["p_did_list"])  # shape: [batch_size]
        index_mapping = batch["index_mapping"]
        enable_hard_neg = "neg_cand_list" in index_mapping
        if enable_hard_neg:
            nc_idx = torch.tensor(batch["nc_dids_list"])  # shape: [batch_size, neg_num]
            nc_idx = nc_idx.view(-1, 1)  # shape: [batch_size * neg_num, 1]
            hard_nc_num = nc_idx.size(0)  # batch_size * neg_num

        ### 训练中不断将可学习温度裁剪在 [1e-3, 0.5] 范围
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        # compute embeddings
        ### 主干编码，得到融合嵌入，形状 [N, embed_dim]，注意 N 是 “query + pos + neg“ 的总扁平条数。
        embeddings = self.encode_multimodal_input(
            txt_dict_batched,
            image_batched,
            txt_mask_batched,
            image_mask_batched,
            use_momentum=False,
        )

        ### 用 index_mapping 从扁平化 embeddings 中取出查询与正例向量并做 L2 归一化。
        # Extract query embeddings
        q_indices = torch.tensor(index_mapping["query"]).flatten()  # shape: [batch_size]
        q_embeds = embeddings[q_indices]  # shape: [batch_size, embed_dim]
        embed_dim = q_embeds.size(1)
        bs = q_embeds.size(0)

        # Extract positive candidate embeddings
        pc_indices = torch.tensor(index_mapping["pos_cand"]).flatten()
        pc_embeds = embeddings[pc_indices]  # shape: [batch_size, embed_dim]

        # normalized features
        q_embeds = F.normalize(q_embeds, dim=-1)
        pc_embeds = F.normalize(pc_embeds, dim=-1)

        # Query Candidate Contrastive Learning
        pc_idx = pc_idx.view(-1, 1)  # [batch_size, 1]

        ### 列表 = [本批正例 ids | (可选)本批负例 ids | 队列中其余 ids]，方便与查询做一对多对齐。
        if enable_hard_neg:
            # If we have hard negatives,
            # we concatenate the positive and negative candidates as well as part of the queue
            idx_all = torch.cat(
                [
                    pc_idx.t().detach(),
                    nc_idx.t().detach(),
                    self.idx_queue.clone()[:, hard_nc_num:].detach(),
                ],
                dim=1,
            )  # [1, batch_size + queue_size]
        else:
            idx_all = torch.cat([pc_idx.t().detach(), self.idx_queue.clone().detach()], dim=1)
            # [1, batch_size + queue_size]

        ### sim_targets 是硬标签
        pos_idx = torch.eq(pc_idx, idx_all).float()  # [batch_size, queue_size + batch_size]
        pre_norm_sim_targets = pos_idx  # [batch_size, queue_size + batch_size]
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)  # [batch_size, queue_size + batch_size]

        ### 动量编码器目标（teacher logits）
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            embeddings_m = self.encode_multimodal_input(
                txt_dict_batched,
                image_batched,
                txt_mask_batched,
                image_mask_batched,
                use_momentum=True,
            )

            # Extract embeddings
            q_embeds_m = embeddings_m[q_indices]  # shape: [batch_size, embed_dim]
            pc_embeds_m = embeddings_m[pc_indices]  # shape: [batch_size, embed_dim]
            nc_embeds_m = None
            if enable_hard_neg:
                nc_indices = torch.tensor(index_mapping["neg_cand_list"])  # shape: [batch_size, neg_num]
                nc_embeds_m = embeddings_m[nc_indices]  # [batch_size, neg_num, embed_dim]

            # Normalized features
            q_embeds_m = F.normalize(q_embeds_m, dim=-1)
            pc_embeds_m = F.normalize(pc_embeds_m, dim=-1)

            # Concatenate with queue
            q_embeds_m_all = torch.cat([q_embeds_m.t(), self.query_queue.clone().detach()], dim=1)

            if enable_hard_neg:
                pc_embeds_m_all = torch.cat(
                    [
                        pc_embeds_m.t(),  # [embed_dim, batch_size]
                        nc_embeds_m.view(hard_nc_num, embed_dim).t().detach(),  # [embed_dim, batch_size * neg_num]
                        self.cand_queue.clone()[:, hard_nc_num:].detach(),
                    ],  # [embed_dim, queue_size]
                    dim=1,
                )
            else:
                pc_embeds_m_all = torch.cat([pc_embeds_m.t(), self.cand_queue.clone().detach()], dim=1)

            # Compute soft labels
            sim_q2pc_m = q_embeds_m @ pc_embeds_m_all / self.temp  # [batch_size, queue_size + batch_size]
            sim_pc2q_m = pc_embeds_m @ q_embeds_m_all / self.temp  # [batch_size, queue_size + batch_size]

            ### 后面那个是重点，最开始是1，逐渐变小，表示从“完全信任 teacher logits”到“完全信任硬标签”。
            sim_q2pc_targets = alpha * F.softmax(sim_q2pc_m, dim=1) + (1 - alpha) * sim_targets
            sim_pc2q_targets = alpha * F.softmax(sim_pc2q_m, dim=1) + (1 - alpha) * sim_targets

        sim_q2pc = q_embeds @ pc_embeds_m_all / self.temp
        sim_pc2q = pc_embeds @ q_embeds_m_all / self.temp

        ### 双向对齐（查询→候选、候选→查询），各自与目标分布做 交叉熵（或 KL） 型损失，最后平均。
        loss_q2pc = -torch.sum(F.log_softmax(sim_q2pc, dim=1) * sim_q2pc_targets, dim=1).mean()
        loss_pc2q = -torch.sum(F.log_softmax(sim_pc2q, dim=1) * sim_pc2q_targets, dim=1).mean()

        loss_contrast = (loss_q2pc + loss_pc2q) / 2

        ### 入队策略与队列更新
        if enable_hard_neg:
            # random chooses to enqueue negative candidates or positive candidates
            enqueue_p = torch.rand(1) < 0.5
            if enqueue_p:
                self._dequeue_and_enqueue(q_embeds_m, pc_embeds_m, pc_idx)
            else:
                nc_idx = nc_idx.view(bs, -1)  # [batch_size, neg_num]
                # We only enqueue the first negative candidate for each query
                self._dequeue_and_enqueue(
                    q_embeds_m,
                    nc_embeds_m[:, 0, :].contiguous(),
                    nc_idx[:, 0].contiguous(),
                )
        else:
            self._dequeue_and_enqueue(q_embeds_m, pc_embeds_m, pc_idx)

        # compute loss and in-batch accuracy
        _max_score, max_idxs = torch.max(sim_q2pc, 1)  # [batch_size]
        predicted_probabilities = pre_norm_sim_targets.gather(1, max_idxs.unsqueeze(1)).squeeze()
        accuracy = predicted_probabilities.mean()
        outputs = {"loss": loss_contrast, "accuracy": accuracy}
        # _, hard_sim_targets_idxs = torch.max(sim_targets, 1)
        # accuracy = (max_idxs == hard_sim_targets_idxs).float().sum() / bs
        # outputs = {"loss": loss_contrast, "accuracy": accuracy}
        return outputs

    def forward(self, batch, alpha=None, encode_mbeir_batch=False):
        if encode_mbeir_batch:
            return self.encode_mbeir_batch(batch)
        return self.compute_contrastive_loss(batch, alpha)

    def encode_mbeir_batch(self, batch):
        # Get id_list
        id_list = batch.get("did_list") or batch.get("qid_list")
        if id_list is None:
            raise ValueError("id_list not found in batch.")

        # Compute embeddings
        embeddings = self.encode_multimodal_input(
            batch["txt_batched"],
            batch["image_batched"],
            batch["txt_mask_batched"],
            batch["image_mask_batched"],
            use_momentum=False,
        )
        assert embeddings.size(0) == len(id_list), "embeddings and id_batched must have the same batch size."
        return embeddings, id_list

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, query_feats, cand_feats, idxs):
        # gather keys before updating queue
        idxs = concat_all_gather(idxs)  # [world_size * batch_size, 1]
        query_feats = concat_all_gather(query_feats)  # [world_size * batch_size, embed_dim]
        cand_feats = concat_all_gather(cand_feats)

        batch_size = query_feats.shape[0]
        ptr = int(self.new_ptr_queue)
        assert self.queue_size % batch_size == 0  # This is important

        # replace the keys at ptr (dequeue and enqueue)
        self.query_queue[:, ptr : ptr + batch_size] = query_feats.T
        self.cand_queue[:, ptr : ptr + batch_size] = cand_feats.T
        self.idx_queue[:, ptr : ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.new_ptr_queue[0] = ptr


def blip_ff(pretrained="", **kwargs):
    model = BLIPFeatureFusion(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
