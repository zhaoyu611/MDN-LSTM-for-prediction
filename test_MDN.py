import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math

NSAMPLE=2500
# x_data=np.float32(np.random.uniform(-10.5, 10.5, (NSAMPLE, 1)))



x_data=np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
r_data=np.float32(np.random.normal(size=(NSAMPLE, 1)))
y_data=np.float32(7.*np.sin(0.75*x_data)+0.5*x_data+r_data)

temp_data=x_data
x_data=y_data
y_data=temp_data

# plt.figure()
# plt.plot(x_data, y_data,'ro', alpha=0.3)


NHIDDEN=24
STDEV=0.5
KMIX=24
NOUT=3*KMIX

x=tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x')
y=tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')

Wh=tf.Variable(tf.random_normal([1,NHIDDEN],stddev=STDEV, dtype=tf.float32))
bh=tf.Variable(tf.random_normal([1, NHIDDEN], stddev=STDEV, dtype=tf.float32))

Wo=tf.Variable(tf.random_normal([NHIDDEN, NOUT],stddev=STDEV, dtype=tf.float32))
bo=tf.Variable(tf.random_normal([1, NOUT], stddev=STDEV, dtype=tf.float32))

hidden_layer=tf.nn.tanh(tf.matmul(x, Wh)+bh)
output=tf.matmul(hidden_layer, Wo)+bo

def get_mixture_coef(output):
    # out_pi=tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name='mixparam')
    # out_sigma=tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name='mixparam')
    # out_mu=tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name='mixparam')

    out_pi, out_sigma, out_mu=tf.split(1, 3, output)
    max_pi=tf.reduce_max(out_pi, 1, keep_dims=True)
    out_pi=tf.sub(out_pi, max_pi)
    out_pi=tf.exp(out_pi)
    normalize_pi=tf.inv(tf.reduce_sum(out_pi,1,keep_dims=True))
    out_pi=tf.mul(normalize_pi, out_pi)

    out_sigma=tf.exp(out_sigma)

    return out_pi, out_sigma, out_mu

out_pi, out_sigma, out_mu=get_mixture_coef(output)

oneDivSqrtTwoPi=1/math.sqrt(2*math.pi) #define a constant used for tf_normal function
def tf_normal(y,mu,sigma):
    result=tf.sub(y, mu)
    result=tf.mul(result, tf.inv(sigma))
    result=-tf.square(result)/2
    return tf.mul(tf.exp(result), tf.inv(sigma)*oneDivSqrtTwoPi)

def get_lossfunc(out_pi, out_sigma, out_mu, y):
    result=tf_normal(y, out_mu, out_sigma)
    result=tf.mul(result, out_pi)
    result=tf.reduce_sum(result,1, keep_dims=True)
    result=-tf.log(result)
    return tf.reduce_mean(result)

lossfunc=get_lossfunc(out_pi, out_sigma, out_mu, y)
train_op=tf.train.AdamOptimizer().minimize(lossfunc)

sess=tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
NEPOCH=10000
loss=np.zeros(NEPOCH) 
# training
for i in range(NEPOCH):
    sess.run(train_op,feed_dict={x: x_data, y: y_data})
    loss[i]=sess.run(lossfunc,feed_dict={x: x_data, y: y_data})

# plt.figure()
# plt.plot(np.arange(100, NEPOCH, 1),loss[100:], 'r-')
# plt.show()



#testing
x_test=np.float32(np.arange(-15, 15, 0.1))
NTEST=x_test.size
x_test=x_test.reshape(NTEST, 1)

def get_pi_idx(x, pdf):
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  print 'error with sampling ensemble'
  return -1

def generate_ensemble(out_pi, out_mu, out_sigma, M = 10):
  NTEST = x_test.size
  result = np.random.rand(NTEST, M) # initially random [0, 1]
  rn = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
  mu = 0
  std = 0
  idx = 0

  # transforms result into random ensembles
  for j in range(0, M):
    for i in range(0, NTEST):
      idx = get_pi_idx(result[i, j], out_pi[i])
      mu = out_mu[i, idx]
      std = out_sigma[i, idx]
      result[i, j] = mu + rn[i, j]*std
  return result

out_pi_test, out_sigma_test, out_mu_test=sess.run(get_mixture_coef(output), feed_dict={x: x_test})
y_test=generate_ensemble(out_pi_test, out_mu_test, out_sigma_test)
plt.figure()
plt.plot(x_data, y_data, 'ro', x_test, y_test, 'bo', alpha=0.3)
# plt.show()

#----draw heatmap------------

x_heat_label=np.float32(np.arange(-15, 15, 0.1))
y_heat_label=np.float32(np.arange(-15, 15, 0.1))

def custom_gaussian(x, mu, std):
    x_norm=(x-mu)/std
    result=oneDivSqrtTwoPi*math.exp(-x_norm*x_norm/2)/std
    return result

def generate_heatmap(out_pi, out_mu, out_sigma, x_heat_label, y_heat_label):
    N=x_heat_label.size
    M=y_heat_label.size
    K=KMIX

    z=np.zeros((N,M))

    mu=0
    std=0
    pi=0

    #transform result into random ensembles
    for k in range(0, K):
        for i in range(0, M):
            pi=out_pi[i, k]
            mu=out_mu[i, k]
            std=out_sigma[i, k]
            for j in range(0, N):
                z[N-j-1, i]+=pi*custom_gaussian(y_heat_label[j], mu, std)
    return z


def draw_heatmap(xedges, yedges, heatmap):
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.figure()
    plt.imshow(heatmap, extent=extent)    
    plt.show()

z=generate_heatmap(out_pi_test, out_mu_test, out_sigma_test, x_heat_label, y_heat_label)

draw_heatmap(x_heat_label, y_heat_label, z)
