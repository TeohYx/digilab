import jax.numpy as np
from jaxtyping import Array, Bool as B, Float as F, UInt16 as U16, jaxtyped
import optax

@jaxtyped
def cross_entropy_loss(
    logits: F[Array, 'bs dst_len n_classes'],
    labels: U16[Array, 'bs dst_len'],
    mask_dec_1d: B[Array, 'bs dst_len'],
) -> F[Array, '']:
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)
    # check_type('loss', loss, F[Array, 'bs dst_len'])
    return np.mean(loss, where=mask_dec_1d)
