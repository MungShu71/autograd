import numpy as np

from classes.Linear import Linear
from classes.autograd import Matrix


class MultiHeadAttention:
    def __init__(self, hidden_size, num_heads):

        self.hidden_size = hidden_size

        self.num_heads = num_heads

        self.d_head = hidden_size // num_heads

        self.query = Linear(hidden_size, hidden_size)
        self.key = Linear(hidden_size, hidden_size)
        self.value = Linear(hidden_size, hidden_size)

        self.proj = Linear(hidden_size, hidden_size)

    def __call__(self, x, flash=False):

        B, N, _ = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.reshape(B, N, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        k = k.reshape(B, N, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        v = v.reshape(B, N, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        
        # batch_size, num_heads, num_patches, d_head
        
        if flash: 
            out = MultiHeadAttention.flash_attention(q,k,v)
        else:

            attn = q.matmul(k.transpose(0, 1, 3, 2))

            attn = attn * (1 / np.sqrt(self.d_head))
            

            attn = attn.softmax()

            out = attn.matmul(v)

            out = out.transpose(0, 2, 1, 3)

            out = out.reshape(B, N, self.hidden_size)

        out = self.proj(out)

        return out
    @staticmethod
    def flash_attention(q, k, v, block_size=16):

        q_mat = q.matrix if isinstance(q, Matrix) else q
        k_mat = k.matrix if isinstance(k, Matrix) else k
        v_mat = v.matrix if isinstance(v, Matrix) else v

        B, H, N, D = q_mat.shape

        scale = 1.0 / np.sqrt(D)

        # output
        o = np.zeros((B, H, N, D), dtype=np.float32)

        # running normalization
        l = np.zeros((B, H, N), dtype=np.float32)

        # running max
        m = np.full((B, H, N), -np.inf, dtype=np.float32)

        # cache for backward
        cache = []

        # =====================================================
        # FORWARD
        # =====================================================

        for j in range(0, N, block_size):

            k_j = k_mat[:, :, j:j + block_size, :]
            v_j = v_mat[:, :, j:j + block_size, :]

            for i in range(0, N, block_size):

                q_i = q_mat[:, :, i:i + block_size, :]
                o_i = o[:, :, i:i + block_size, :]
                l_i = l[:, :, i:i + block_size]
                m_i = m[:, :, i:i + block_size]

                s_ij = (q_i @ k_j.transpose(0, 1, 3, 2)) * scale

                m_ij = np.max(s_ij, axis=-1)

                p_ij = np.exp(s_ij - m_ij[:, :, :, None])

                l_ij = np.sum(p_ij, axis=-1)

                m_new = np.maximum(m_i, m_ij)

                alpha = np.exp(m_i - m_new)

                beta = np.exp(m_ij - m_new)

                l_new = alpha * l_i + beta * l_ij

                numer = (
                    alpha[:, :, :, None] * l_i[:, :, :, None] * o_i
                    + beta[:, :, :, None] * (p_ij @ v_j)
                )

                o[:, :, i:i + block_size, :] = numer / l_new[:, :, :, None]

                l[:, :, i:i + block_size] = l_new

                m[:, :, i:i + block_size] = m_new

                # save EVERYTHING needed for backward
                cache.append(
                    (
                        i,
                        j,
                        q_i.copy(),
                        k_j.copy(),
                        v_j.copy(),
                        p_ij.copy(),
                        m_ij.copy(),
                    )
                )

        out_matrix = o.transpose(0, 2, 1, 3).reshape(B, N, H * D)

        out = Matrix(out_matrix, (q, k, v))

        # =====================================================
        # BACKWARD
        # =====================================================

        def _backward():

            dout = (
                out.grad
                .reshape(B, N, H, D)
                .transpose(0, 2, 1, 3)
            )

            dq = np.zeros_like(q_mat)
            dk = np.zeros_like(k_mat)
            dv = np.zeros_like(v_mat)

            for (
                i,
                j,
                q_i,
                k_j,
                v_j,
                p_ij,
                m_ij,
            ) in cache:

                do_i = dout[:, :, i:i + block_size, :]

                # final normalization for this row
                l_final = l[:, :, i:i + block_size]

                # final max for this row
                m_final = m[:, :, i:i + block_size]

                # reconstruct true softmax probabilities
                P = (
                    p_ij
                    * np.exp(m_ij - m_final)[:, :, :, None]
                ) / l_final[:, :, :, None]

                # dV
                dv[:, :, j:j + block_size, :] += (
                    P.transpose(0, 1, 3, 2) @ do_i
                )

                # output block
                o_i = o[:, :, i:i + block_size, :]

                # softmax correction term
                D_i = np.sum(do_i * o_i, axis=-1, keepdims=True)

                # dP = dO @ V^T
                dP = do_i @ v_j.transpose(0, 1, 3, 2)

                # softmax backward
                dS = P * (dP - D_i)

                dS *= scale

                # dQ
                dq[:, :, i:i + block_size, :] += dS @ k_j

                # dK
                dk[:, :, j:j + block_size, :] += (
                    dS.transpose(0, 1, 3, 2) @ q_i
                )

            q.grad += dq
            k.grad += dk
            v.grad += dv

        out._backward = _backward
        return out   

    def parameters(self):

        return (
            self.query.parameters()
            + self.key.parameters()
            + self.value.parameters()
            + self.proj.parameters()
        )