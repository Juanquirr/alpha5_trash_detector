import os
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    save_attention_maps
)


class AttentionMapExtractor:
    """
    Extracts and aggregates cross‐attention maps per token from a diffusion model’s internal attn_maps.
    """

    def __init__(self, attn_maps, tokenizer, base_dir='attn_maps', unconditional=True):
        """
        attn_maps: nested dict timestep → layer → tensor(batch, n_heads, H, W, seq_len)
        tokenizer: a HuggingFace‐style tokenizer
        base_dir:  directory to dump per‐timestep maps (optional)
        unconditional: whether to discard the “unconditional” half of classifier‐free split
        """
        self.attn_maps = attn_maps
        self.tokenizer = tokenizer
        self.base_dir = base_dir
        self.uncond = unconditional
        self.to_pil = ToPILImage()

    def compute(self, prompts):
        """
        prompts: list of text prompts
        Returns: dict token → aggregated attention map tensor (H×W)
        """
        # 1) tokenize once
        input_ids = self.tokenizer(prompts, return_tensors='pt', padding=True)['input_ids']
        # ensure list of lists
        batches = input_ids if input_ids.ndim == 2 else input_ids.unsqueeze(0)
        tokens_per_batch = [self.tokenizer.convert_ids_to_tokens(b.tolist()) for b in batches]

        # 2) accumulate resized maps
        # pick shape from first layer
        sample = next(iter(next(iter(self.attn_maps.values())).values()))
        b, h, H, W, seq = sample.shape

        # sum over heads and optionally drop unconditional
        def prepare(am):
            # am: (b,heads,H,W,seq)
            m = am.sum(1)  # (b,H,W,seq)
            if self.uncond:
                m = m.chunk(2, dim=0)[1]
            return m.permute(0, 3, 1, 2)  # (batch,seq,H,W)

        # init accumulator
        first = prepare(sample)
        agg = torch.zeros_like(first)
        count = 0
        for tstep, layer_dict in self.attn_maps.items():
            os.makedirs(os.path.join(self.base_dir, str(tstep)), exist_ok=True)
            for layer_map in layer_dict.values():
                m = prepare(layer_map)
                # resize to match first
                m = F.interpolate(m, size=first.shape[-2:], mode='bilinear', align_corners=False)
                agg += m
                count += 1
        agg /= count  # average

        # 3) split per token
        result = {}
        for batch_idx, tokens in enumerate(tokens_per_batch):
            for tok_idx, tok in enumerate(tokens):
                # ensure unique key
                key = tok
                if key in result:
                    i = 1
                    while f"{tok}_{i}" in result:
                        i += 1
                    key = f"{tok}_{i}"
                result[key] = agg[batch_idx, tok_idx]
        return result


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def plot_attention_maps(images, avg_maps, cols=3, save_path=None):
    """
    Plots original image + attention maps in a grid with 'cols' columns.

    Args:
        images: list or array of original images (expects at least one).
        avg_maps: dict with keys as labels and values as numpy arrays (attention maps).
        cols: int, number of columns per row in the grid (default 3).
        save_path: str or None, if set saves the figure to this path.

    Usage:
        plot_attention_maps(images, avg_maps, cols=4)
    """
    keys = sorted(avg_maps)  # keys order as inserted; change to sorted(avg_maps) for alphabetical

    n_maps = len(keys) + 1  # +1 for original image
    rows = (n_maps + cols - 1) // cols  # ceiling division for rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if n_maps > 1 else [axes]

    # Plot original image
    axes[0].imshow(images[0])
    axes[0].set_title("Imagen original", fontsize=12, pad=8)
    axes[0].axis("off")

    # Plot attention maps
    for i, key in enumerate(keys, 1):
        axes[i].imshow(avg_maps[key])
        axes[i].set_title(str(key), fontsize=12, pad=8)
        axes[i].axis("off")

    # Turn off unused subplots
    for j in range(n_maps, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_heatmap_3d_side_by_side(heatmap):
    """
    Plots a 3D heatmap from multiple elevation angles side by side.

    Args:
        heatmap: 2D numpy array representing the heatmap.

    Usage:
        plot_heatmap_3d_side_by_side(avg_maps['horse</w>'].detach().numpy())
    """
    heatmap = heatmap.astype(float)
    H, W = heatmap.shape
    x = np.arange(W)
    y = np.arange(H)
    x, y = np.meshgrid(x, y)

    angles = [
        (0, 90),  # (elev, azim)
        (15, 90),
        (30, 90),
        (45, 90),
        (60, 90),
        (75, 90),
        (90, 90),
    ]

    fig = plt.figure(figsize=(25, 7))
    for i, (elev, azim) in enumerate(angles, 1):
        ax = fig.add_subplot(1, len(angles), i, projection='3d')
        ax.plot_surface(x, y, heatmap, cmap='viridis')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Elev={elev}°, Azim={azim}°", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.tight_layout()
    plt.show()


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


class AttentionMixer:
    def __init__(self, pipe, exclude_tokens=None):
        self.pipe = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        # Tokens a excluir por defecto
        if exclude_tokens is None:
            exclude_tokens = ["<|endoftext|>", "<|startoftext|>"]
        self.exclude_tokens = set(exclude_tokens)

    def _get_embeddings_and_tokens(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        input_ids = inputs.input_ids.to(self.pipe.device)
        embeddings = self.text_encoder(input_ids)[0].squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        valid = [(i, tok) for i, tok in enumerate(tokens) if
                 not tok.startswith("<") and
                 tok != "</w>" and
                 tok not in self.exclude_tokens]
        if len(valid) == 0:
            return embeddings, tokens
        indices, clean_tokens = zip(*valid)
        return embeddings[list(indices)], list(clean_tokens)

    def _get_negative_words(self, prompt_tokens, positive_words):
        # Excluir también tokens que estén en exclude_tokens
        return [tok for tok in prompt_tokens if tok not in positive_words and tok not in self.exclude_tokens]

    def compute_map(self, prompt, positive_words, negative_words=None, avg_maps=None,
                    gamma=1.0, visualize_attention=False, plot_relation=False):
        if avg_maps is None:
            raise ValueError("Debe proporcionar avg_maps con mapas de atención por token.")

        prompt_embs, prompt_tokens = self._get_embeddings_and_tokens(prompt)

        if negative_words is None:
            negative_words = self._get_negative_words(prompt_tokens, positive_words)

        pos_phrase = " ".join(positive_words)
        pos_embs, pos_tokens = self._get_embeddings_and_tokens(pos_phrase)

        neg_phrase = " ".join(negative_words) if negative_words else ""
        neg_embs, neg_tokens = self._get_embeddings_and_tokens(neg_phrase) if negative_words else (None, [])

        Q = prompt_embs
        if negative_words:
            K = torch.cat([pos_embs, neg_embs], dim=0)
            all_tokens = pos_tokens + neg_tokens
        else:
            K = pos_embs
            all_tokens = pos_tokens

        attn_scores = Q @ K.T
        attn_weights = F.softmax(attn_scores, dim=0)

        final_map = None
        total_weight = 0.0

        for i_tok_prompt, tok_prompt in enumerate(prompt_tokens):
            for i_tok_target, tok_target in enumerate(all_tokens):
                weight = attn_weights[i_tok_prompt, i_tok_target].item()

                if tok_target not in avg_maps:
                    continue

                if tok_target in positive_words:
                    sign = 1
                elif negative_words and tok_target in negative_words:
                    sign = -1
                else:
                    sign = 0

                if sign == 0:
                    continue

                weighted_map = sign * weight * torch.tensor(avg_maps[tok_target])
                final_map = weighted_map if final_map is None else final_map + weighted_map
                total_weight += abs(weight)

        if final_map is None:
            print("❌ No se encontraron mapas útiles.")
            return None

        combined_map = (final_map / (total_weight + 1e-8)).numpy()
        normalized = combined_map / (combined_map.max() + 1e-8)
        gamma_corrected = normalized ** gamma

        return gamma_corrected

    def plot_word_attention_relations(self, tokens_left, tokens_right, attn_matrix):
        """
        Dibuja tokens_left y tokens_right en lados opuestos,
        con líneas cuya opacidad y grosor representan la atención entre ellos.
        """
        fig, ax = plt.subplots(figsize=(10, max(len(tokens_left), len(tokens_right)) * 0.5))

        y_left = np.linspace(0, 1, len(tokens_left))
        y_right = np.linspace(0, 1, len(tokens_right))

        x_left = 0.1
        x_right = 0.9

        # Escribir tokens
        for i, word in enumerate(tokens_left):
            ax.text(x_left - 0.05, y_left[i], word, ha='right', va='center', fontsize=12)
        for j, word in enumerate(tokens_right):
            ax.text(x_right + 0.05, y_right[j], word, ha='left', va='center', fontsize=12)

        # Normalizar matriz para usar en grosor y alpha
        attn_norm = attn_matrix / attn_matrix.max()

        # Dibujar líneas
        for i, y_l in enumerate(y_left):
            for j, y_r in enumerate(y_right):
                weight = attn_norm[i, j]
                if weight > 0.05:
                    ax.plot([x_left, x_right], [y_l, y_r],
                            lw=weight * 5, alpha=weight, color='blue')

        ax.axis('off')
        ax.set_title("Relación de atención entre palabras", fontsize=14)
        plt.show()

    def plot_word_attention_relations(self, tokens_left, tokens_right, attn_matrix):
        """
        Dibuja tokens_left y tokens_right en lados opuestos,
        con líneas cuya opacidad y grosor representan la atención entre ellos.
        """
        fig, ax = plt.subplots(figsize=(10, max(len(tokens_left), len(tokens_right)) * 0.5))

        y_left = np.linspace(0, 1, len(tokens_left))
        y_right = np.linspace(0, 1, len(tokens_right))

        x_left = 0.1
        x_right = 0.9

        # Escribir tokens
        for i, word in enumerate(tokens_left):
            ax.text(x_left - 0.05, y_left[i], word, ha='right', va='center', fontsize=12)
        for j, word in enumerate(tokens_right):
            ax.text(x_right + 0.05, y_right[j], word, ha='left', va='center', fontsize=12)

        # Normalizar matriz para usar en grosor y alpha
        attn_norm = attn_matrix / attn_matrix.max()

        # Dibujar líneas
        for i, y_l in enumerate(y_left):
            for j, y_r in enumerate(y_right):
                weight = attn_norm[i, j]
                if weight > 0.05:
                    ax.plot([x_left, x_right], [y_l, y_r],
                            lw=weight * 5, alpha=weight, color='blue')

        ax.axis('off')
        ax.set_title("Relación de atención entre palabras", fontsize=14)
        plt.show()


import torch
import re

def compute_avg_map(word: str, maps: dict) -> torch.Tensor:
    """
    Computes the average attention map for a given word by matching it to token keys in `maps`.

    Args:
        word: The word to match (e.g. "city", "sailboat").
        maps: Dict[str, np.ndarray or torch.Tensor] where keys are token strings.

    Returns:
        torch.Tensor: The averaged attention map over matching tokens.
    """
    word_lower = word.lower()
    matched_maps = []

    for token, tensor in maps.items():
        # Remove _N suffixes like _1, _2
        token_clean = re.sub(r"_\d+$", "", token)
        # Strip BPE markers like </w>
        token_clean = token_clean.replace("</w>", "")
        token_clean = token_clean.lower()

        if token_clean in word_lower or word_lower in token_clean:
            matched_maps.append(torch.from_numpy(tensor) if isinstance(tensor, (np.ndarray,)) else tensor)

    if not matched_maps:
        raise ValueError(f"No matching tokens found for word '{word}' in attention map keys.")

    return torch.stack(matched_maps).mean(dim=0)

import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import label


def extract_refined_boxes(end_map, image, refine_bounding_box_fn, gamma=2.0):
    """
    Extrae y refina bounding boxes desde un mapa de finalización (end_map) usando clustering KMeans y una función de refinamiento (SAM).

    Args:
        end_map (np.ndarray or torch.Tensor): Mapa de calor/end_map 2D.
        image (PIL.Image): Imagen original correspondiente.
        refine_bounding_box_fn (function): Función de refinamiento de bounding box (SAM).
        gamma (float): Valor de corrección gamma (default = 2.0).

    Returns:
        List[List[int]]: Lista de cajas refinadas en formato [x_min, y_min, x_max, y_max].
    """
    # Convertir a numpy si es tensor
    end_map_np = end_map.cpu().numpy() if not isinstance(end_map, np.ndarray) else end_map

    # Normalización y corrección gamma
    end_map_norm = (end_map_np - np.min(end_map_np)) / (np.max(end_map_np) - np.min(end_map_np) + 1e-8)
    end_map_gamma = end_map_norm ** gamma

    # Clustering con KMeans
    H_em, W_em = end_map_gamma.shape
    X = end_map_gamma.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    labels = kmeans.labels_.reshape(H_em, W_em)

    centers = kmeans.cluster_centers_.flatten()
    obj_cluster = centers.argmax()
    mask_obj = (labels == obj_cluster)

    # Regiones conectadas
    labeled_mask, num_features = label(mask_obj)

    # Escala a la imagen original
    W_img, H_img = image.size
    scale_x = W_img / W_em
    scale_y = H_img / H_em

    refined_boxes = []

    for region_label in range(1, num_features + 1):
        region = (labeled_mask == region_label)
        coords = np.column_stack(np.where(region))
        if coords.size == 0:
            continue

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        x_min_s = int(x_min * scale_x)
        x_max_s = int(x_max * scale_x)
        y_min_s = int(y_min * scale_y)
        y_max_s = int(y_max * scale_y)

        box = [x_min_s, y_min_s, x_max_s, y_max_s]

        # Refinar con SAM
        refined = refine_bounding_box_fn(image, box)
        if refined and 'bounding_boxes' in refined and refined['bounding_boxes']:
            b = refined['bounding_boxes'][0]
            refined_box = [b['x_0'], b['y_0'], b['x_1'], b['y_1']]
        else:
            refined_box = box

        refined_boxes.append(refined_box)

    return refined_boxes
