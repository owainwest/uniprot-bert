import statistics

acid_to_pks = {
    'a': [9.87,2.35],
    'r': [9.09,2.18],
    'n': [8.8,2.02],
    'd': [9.6,1.88],
    'c': [10.78,1.71],
    'e': [9.67,2.19],
    'q': [9.13,2.17],
    'g': [9.6,2.34],
    'h': [8.97,1.78],
    'i': [9.76,2.32],
    'l': [9.6,2.36],
    'k': [10.28,8.9],
    'm': [9.21,2.28],
    'f': [9.24,2.58],
    'p': [10.6,1.99],
    's': [9.15,2.21],
    't': [9.12,2.15],
    'w': [9.39,2.38],
    'y': [9.11,2.2],
    'v': [9.72,2.29]        
}
DEFAULT_GUESS = statistics.median(sum(v) for v in acid_to_pks.values())
print(min(sum(v) for v in acid_to_pks.values()))
print(max(sum(v) for v in acid_to_pks.values()))
print(DEFAULT_GUESS)










# import tensorflow as tf

# # values of x = [1,2,3,4]
# # values of y = [0,-1,-2,-3]

# x_train = [1,2,3,4]
# y_train = [0,-1,-2,-3]

# #defining the weight and bias
# W = tf.Variable([-.5], dtype=tf.float32)
# b = tf.Variable([.5], dtype=tf.float32)

# x = tf.placeholder(dtype=tf.float32)
# y = tf.placeholder(dtype=tf.float32) 

# # using linear function y = Wx + b
# lm = W*x + b

# #calculating squared error
# loss = tf.reduce_sum(tf.square(lm - y))

# #using Gradient Descent with learning rate 0.01
# optimizer = tf.train.GradientDescentOptimizer(0.01)

# #minimizing loss
# train = optimizer.minimize(loss)

# session = tf.Session()
# init = tf.global_variables_initializer()
# session.run(init)

# #training model for 1000 iterations
# for i in range(1000):
#     session.run(train, {x:x_train, y:y_train})

# #final values of W and b
# print(session.run([W,b]))
# #output of the model
# print(session.run(lm,{x:[5,6,7,8]}))