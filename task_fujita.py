import math
import random

TEACH_NUM = 4
INP_NUM = 2
theta = 0
d_theta = 0
Alpha = 0.01

w = [0,0]
dw = [0,0]

teach_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
teach_y = [0, 1, 0, 1]

r = random.random()

def sigmoid(x):
  return 1.0 / (1.0 + math.exp(-x))

def forward(x):
  u = 0
  for i in range(INP_NUM):
    u += w[i] * x[i]
  u += theta
  return sigmoid(u)

def func_error():
  e = 0
  for i in range(TEACH_NUM):
    y = forward(teach_x[i])
    e += 0.5 * (y - teach_y[i])**2
  return e

def clear_dw():
  global dw, d_theta
  for i in range(INP_NUM):
    dw[i] = 0
  d_theta = 0

def calc_dw(x_t, y_hat):
  global dw, d_theta
  y = forward(x_t)
  dy = y * (1 - y)
  for i in range(INP_NUM):
    dw[i] += (y - y_hat) * dy * x_t[i]
  d_theta += (y - y_hat) * dy

def init_w():
  global w, theta
  for i in range(INP_NUM):
    w[i] = r * 2 - 1.0
  theta = r * 2 - 1.0

def update_w():
  global w, theta
  for i in range(INP_NUM):
    w[i] -= Alpha * dw[i]
  theta -= Alpha * d_theta

if __name__ == '__main__':   
  init_w()
  
  for loop in range(100000):
    if loop % 1000 == 0:
      print(str(loop) + ', ' + str(func_error()))
    clear_dw()
    
    for j in range(TEACH_NUM):
      calc_dw(teach_x[j], teach_y[j])

    update_w()
  print(str(loop) + ', ' + str(func_error()))

  for i in range(TEACH_NUM):
    y = forward(teach_x[i])
    print(str(i) + ': y = ' + str(y) + ' <--> y_hat = ' + str(teach_y[i]))