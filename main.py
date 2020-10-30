ALPHA = 1
THETA = 0.2
AND_INPUTS = [
        {
            "data": (1, 1, 1), "target": 1
        },
        {
            "data": (1, 0, 1), "target": -1
        },
        {
            "data": (0, 1, 1), "target": -1
        },
        {
            "data": (0, 0, 1), "target": -1
        },
]

OR_INPUTS = [
        {
            "data": (1, 1, 1), "target": 1
        },
        {
            "data": (1, 0, 1), "target": 1
        },
        {
            "data": (0, 1, 1), "target": 1
        },
        {
            "data": (0, 0, 1), "target": -1
        },
]

def dot_product(dot_inputs, dot_weights):
    return \
    dot_inputs[0] * dot_weights[0] + \
    dot_inputs[1] * dot_weights[1] + \
    dot_inputs[2] * dot_weights[2]

def calc_weight_delta(target, x):
    delta_w = [0, 0, 0]

    for i in range(len(delta_w)):
        delta_w[i] = ALPHA * target * x[i]

    return delta_w

def step_fn(y_in):
    if y_in > THETA:
        return 1
    elif -THETA <= y_in <= THETA:
        return 0
    else:
        return -1

def weights_is_all_zero(matrix):
    for row in matrix:
        for el in row:
            if el != 0:
                return False

    return True

def train(weights, inputs):
    epoch = 1
    done = False


    for i in range(1, 11):
        print(f'Starting epoch {epoch} with weights {weights}...')
        print('')

        weight_delta_vec = []

        print('| x1 | x2 | b | y-in          | y  | target | Δw1 | Δw2 | Δbi | w1(0) | w2(0) | b(0) |')
        print('|----|----|---|---------------|----|:------:|:---:|:---:|:---:|:-----:|:-----:|:----:|')

        for inp in inputs:
            x = inp.get("data")
            target = inp.get("target")

            y_in = dot_product(x, weights)
            y = step_fn(y_in)

            weight_deltas = [0, 0, 0]
            if y != target:
                weight_deltas = calc_weight_delta(target, x)

                for i in range(len(weights)):
                    weights[i] += weight_deltas[i]

            weight_delta_vec.append(weight_deltas)

            to_print = f'| {x[0]} | {x[1]} | {x[2]} | {y_in} | {y} | {target} | {weight_deltas[0]} |'
            to_print += f'{weight_deltas[1]} | {weight_deltas[2]} | {weights[0]} | {weights[1]} |'
            to_print += f'{weights[2]} |'
            print(to_print)

        print('')
        done = weights_is_all_zero(weight_delta_vec)

        if done:
            return (epoch, weights)

        epoch += 1

def predict(weights, inputs):
    y_in = dot_product(inputs, weights)
    return step_fn(y_in)

if(__name__ == "__main__"):
    weights = [0, 0, 0]
    inputs = AND_INPUTS
    result = train(weights, inputs)

    print(f'Training done with epoch count: {result[0]}\n')
    print(f'Training done with weights: {result[1]}\n')

    prediction = predict(weights, (1, 0, 1))
    print(f'Predict with (1, 0, 0): {prediction}\n')

    prediction = predict(weights, (0, 0, 1))
    print(f'Predict with (0, 0, 0): {prediction}\n')

