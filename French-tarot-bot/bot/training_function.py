import tensorflow as tf
from tensorflow.keras import optimizers


def train_network_ppo(model, states, actions, available_actions, values, a1, a2, eps, lr, batch_size, n_epochs, clip_norm=1.0, freq=1.0):
    """
    Train the model with the PPO loss

    Parameters
    ----------
    model : keras functional
        Neural network that will be optimized
    states : array
        32-bit float array composing the states history
    actions : array
        32-bit float array composing the history of played moves
    available_actions : array
        32-bit float array composing the history of available moves
    values : array
        32-bit float array composing the history of values
    a1 : float
        PPO coefficient of the value loss
    a2 : float
        PPO coefficient of the entropy loss
    eps : float
        Smoothing coefficient
    lr : float
        Learning rate
    batch_size : int
        Batch size
    n_epochs : int
        Number of epochs
    clip_norm : float
        Add a clipping norm to the optimizer
    freq : float
        Proportion of history used

    Returns
    ----------
    clip_loss : float
        Last iteration average clip loss
    value_loss : float
        Last iteration average value loss
    entropy_loss : float
        Last iteration average entropy loss
    """
    n_games = states.shape[0]
    optimizer = optimizers.Adam(learning_rate=lr, clipnorm=clip_norm)

    for _ in range(n_epochs):
        # Training loop
        random_shuffle = tf.range(n_games)
        tf.random.shuffle(random_shuffle)
        # Shuffle the moves
        shuffle_states = states[random_shuffle]
        shuffle_actions = actions[random_shuffle]
        shuffle_available_actions = available_actions[random_shuffle]
        shuffle_values = values[random_shuffle]

        n_iterations = int(freq*(n_games//batch_size))
        # Number of iterations based on batch size and frequency
        for i in range(n_iterations):
            selected_moves = tf.range(batch_size*i, batch_size*(i+1))
            # Indices of the moves for which the PPO will be applied
            selected_states = shuffle_states[selected_moves]
            selected_actions = shuffle_actions[selected_moves]
            selected_available_actions = shuffle_available_actions[selected_moves]
            selected_values = shuffle_values[selected_moves]
            # Select the indices
            fixed_policy, fixed_values = model(
                [selected_states, selected_available_actions])
            selected_advantages = selected_values - fixed_values
            # Compute the fixe advantages

            delta = 1e-9
            # Prevents division by zero

            with tf.GradientTape() as tape:
                expected_policy, expected_values = model(
                    [selected_states, selected_available_actions])
                # Compute the optimizable values

                rt = tf.reduce_sum(
                    selected_actions*expected_policy/(fixed_policy+delta), axis=1, keepdims=True)

                clip_loss = tf.reduce_mean(
                    tf.minimum(
                        tf.multiply(rt, selected_advantages),
                        tf.multiply(tf.clip_by_value(
                            rt, 1-eps, 1+eps), selected_advantages)
                    ), axis=1, keepdims=True)

                value_loss = tf.square(expected_values-selected_values)

                entropy_loss = tf.reduce_sum(-tf.multiply(tf.math.log(
                    expected_policy+delta), expected_policy), axis=1, keepdims=True)

                total_loss = - \
                    tf.reduce_mean(clip_loss-a1*value_loss+a2*entropy_loss)

            grads = tape.gradient(total_loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return clip_loss, value_loss, entropy_loss
