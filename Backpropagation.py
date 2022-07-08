import math

input_matrix =      [0.9, 0.1, 0.8]
w_input_hidden =    [[0.9, 0.3, 0.4],
                    [0.2, 0.8, 0.2],
                    [0.1, 0.5, 0.6]]
w_hidden_output =   [[0.3, 0.7, 0.5],
                    [0.6, 0.5, 0.2],
                    [0.8, 0.1, 0.9]]
target =            [0.6, 0.8, 0.5]
x_hidden = []
o_hidden = []       
x_output = []  
o_output = [] 

# y = sigmoid(x)
def sigmoid(x):
    y = [0] * len(x)
    for i in range(len(x)):
        sig_funct = 1/(1 + math.exp((-1) * x[i]))
        # format the result to only 3 digits after decimal points
        format_sig_funct = "{:.8f}".format(sig_funct)
        y[i] = (float(format_sig_funct))
    return y

# z = x * y
def matrix_mul(x, y):
    z = [0] * len(x)
    for i in range(len(x)):
        element = 0
        for j in range(len(y)):
            element =  element + (x[i][j] * y[j])
        z[i] = element
    return z


# FORWARD
# find matrix X_hidden
x_hidden = matrix_mul(w_input_hidden, input_matrix)

# find matrix O_hidden
o_hidden = sigmoid(x_hidden)

# find matrix X_output
x_output = matrix_mul(w_hidden_output, o_hidden)

# find matrix O_output
o_output = sigmoid(x_output)


#BACKPROPAGATION
o_output_new = o_output
dE_total = [0]*3
dO_output = [0]*3
dX_output_w = [0]*3
dE_total_w_hidden_output = [0]*3
w_hidden_output_update = w_hidden_output
dO_hidden = [0]*3
dE_total_x_hidden = [0]*3
dE_total_w_input_hidden = [0]*3
w_input_hidden_update = w_input_hidden
x_hidden_new = x_hidden
o_hidden_new = o_hidden
x_output_new = x_output

count = 0
while count < 295:
    # find derivative of total error
    for i in range(len(target)):
        temp = (-1 * target[i]) + o_output_new[i]
        print(temp)
        dE_total[i] = temp

    # find derivative of O_output
    for i in range(len(o_output_new)):
        temp = o_output_new[i] - (o_output_new[i])**2
        dO_output[i] = temp

    # find derivative of X_output_w
    for i in range(len(o_hidden)):
        temp = o_hidden[i]
        dX_output_w[i] = temp

    # find derivative of E_total_w_hidden_output
    for i in range(len(dE_total)):
        temp = dE_total[i] * dO_output[i] * dX_output_w[i]
        dE_total_w_hidden_output[i] = temp

    # find updated w_hidden_output
    # let learning rate alpha be 0.5
    alpha = 0.5
    for i in range(len(w_hidden_output)):
        temp = [0]*3
        element = 0
        for j in range(len(w_hidden_output[i])):
            element = w_hidden_output[i][j] - (alpha * dE_total_w_hidden_output[i])
            temp[j] = element
        w_hidden_output_update[i] = temp

    # find derivative of O_hidden
    for i in range(len(o_hidden)):
        temp = o_hidden[i] - (o_hidden[i])**2
        dO_hidden[i] = temp

    # find derivative of E_total_x_hidden
    for i in range(len(w_hidden_output)):
        temp = dE_total[i] * dO_output[i] * w_hidden_output[i][0]
        dE_total_x_hidden[i] = temp

    # find derivative of E_total_w_input_hidden
    for i in range(len(input_matrix)):
        temp = dE_total_x_hidden[i] * dO_hidden[i] * input_matrix[i]
        dE_total_w_input_hidden[i] = temp

    # find updated w_input_hidden
    # let learning rate alpha be 0.5
    alpha = 0.5
    for i in range(len(w_input_hidden)):
        temp = [0]*3
        element = 0
        for j in range(len(w_input_hidden[i])):
            element = w_input_hidden[i][j] - (alpha * dE_total_w_input_hidden[i])
            temp[j] = element
        w_input_hidden_update[i] = temp


    # FORWARD TO CHECK
    x_hidden_new = matrix_mul(w_input_hidden_update, input_matrix)
    o_hidden_new = sigmoid(x_hidden_new)
    x_output_new = matrix_mul(w_hidden_output_update, o_hidden_new)
    o_output_new = sigmoid(x_output_new)

    count += 1

print("Input : " + str(input_matrix))
print("Output : " + str(o_output))
print("Ater training output : " + str(o_output_new))