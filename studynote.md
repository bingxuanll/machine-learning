# 第一部分:回归和分类问题
## 监督学习
- 回归问题 根据所给数据集拟合出适合的函数从而进行数值预测
- 分类问题 通过所给数据集学习后可以预测输入对应的类别
## 无监督学习
- 聚类问题
## 线性回归模型 f=wx+b

### 代价函数J
>全部m个样本损失的平均值
 
  $$
  J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}
  $$ 

其中：

$$
f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{2}
$$

- **单特征值x = np.array([ ]),w = _**
```python
def compute_cost(x,y,w,b):
    #x(ndarray(m,))
    #y(ndarray(m,))
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb = w*x[i] + b
        cost += (f_wb - y[i])**2
    cost = (1/(2*m))*cost
    return cost
```
- **多特征值x = np.array([],[],[]....),w = [ , ,...]  -->  w.shape[0] == x.shape[1]**
```python
def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb = np.dot(x[i],w) + b #dot(x,w) 实现w和x的点积
        cost = cost + (f_wb - y[i])**2
    cost = cost / (2*m)
    return cost
```
### 梯度下降
>沿着代价函数J梯度的反方向，一点点更新参数w,b，不断减小代价函数，直到收敛

- **单特征值x = np.array([ ]),w = _**  
  
$$
\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline
\;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{3}  \; \newline 
 b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
\end{align*}
$$

其中：

$$
\begin{align}
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \tag{4}\\
  \frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \tag{5}\\
\end{align}
$$

```python
#求偏导
def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0.
    dj_db = 0.
    for i in range(m):
        f_wb = w*x[i]+b
        dj_dw_i = (f_wb-y[i])*x[i]
        dj_db_i = f_wb-y[i]
        dj_dw += dj_dw_i
        dj_db += dj_wb_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw,dj_db

#梯度下降函数
import copy
def gradient_descent(x,y,w_in,b_in,alpha,compute_gradient,compute_cost,num_iters):
    w = copy.deepcopy(w_in) #deepcopy() 形成w的深层拷贝
    J_his = [] #存放历史代价J
    P_his = [] #存放历史参数[w,b]
    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw,dj_db = compute_gradient(x,y,w,b)
        #update w,b
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        #save J and w,b
        if i < 100000:
            J_his.append(compute_cost(x,y,w,b))
            P_his.append([w,b])
        
        if i%math.ceil(num_iters/10)==0:
            print(f"Iteration {i:4}:Cost {J_his[-1]:0.2e}",
                 f"dj_dw:{dj_dw:0.3e},dj_db:{dj_wb:0.3e}",
                 f"w:{w:0.3e},b:{b:0.3e}")
        

    return w,b,J_his,P_his
```
- **多特征值x = np.array([],[],[]....),w = [ , ,...]  -->  w.shape[0] == x.shape[1]**
  
$$
\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline\;
& w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{5}  \; & \text{for j = 0..n-1}\newline
&b\ \ = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}  \newline \rbrace
\end{align*}
$$

其中：

$$
\begin{align}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{6}  \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{7}
\end{align}
$$

```python
#偏导 
def compute_gradient(x,y,w,b):
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(x[i],w)+b)-y[i]
        for j in range(n):
            dj_dw[j] += err * x[i,j]
        dj_db += err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw,dj_db

#梯度下降函数
def gradient_descent(x,y,w,b,alpha,compute_cost,compute_gradient,num_iters):
    w = copy.deepcopy(w)
    J_his = []
    b = b

    for i in range(num_iters):
        dj_dw,dj_db = compute_gradient(x,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            J_his.append(compute_cost(x,y,w,b))
        if i%math.ceil(num_iters/10)==0:
            print(f"Iteration:{i:4d}:Cost{J_his[-1]:8.2f}")
    return w,b,J_his
```
### 学习率 alpha
![学习率](md-images/learningrate.PNG)

1. **alpha合适：** 代价平滑持续下降
2. **alpha太小：** 代价下降缓慢，需要迭代很多次才能收敛，训练很慢
3. **alpha太大：** 代价震荡，来回跳动，甚至导致越来越大，不收敛
### Z-score标准化
属于特征标准化，特征化之后该特征均值=0，标准差=1，用于消除不同特征量级差异从而使收敛速度大幅变快。

$$
x^{(i)}_j = \dfrac{x^{(i)}_j - \mu_j}{\sigma_j} \tag{4}
$$ 

其中：

$$
\begin{align}
\mu_j &= \frac{1}{m} \sum_{i=0}^{m-1} x^{(i)}_j \tag{5}\\
\sigma^2_j &= \frac{1}{m} \sum_{i=0}^{m-1} (x^{(i)}_j - \mu_j)^2  \tag{6}
\end{align}
$$

```python- [监督学习](#监督学习)
- [无监督学习](#无监督学习)
- [线性回归模型 f=wx+b](#线性回归模型-fwxb)
  - [代价函数J](#代价函数j)
  - [梯度下降](#梯度下降)
  - [学习率 alpha](#学习率-alpha)
  - [Z-score标准化](#z-score标准化)
  - [多项式回归](#多项式回归)
- [逻辑回归](#逻辑回归)
  - [sigmoid函数](#sigmoid函数)
  - [损失函数Loss](#损失函数loss)

def Z_score(x):
    #按列求平均值
    mu = np.mean(x,axis = 0)
    #求标准差
    sigma = np.std(x,axis = 0)
    x_norm = (x - mu)/sigma
    return mu,sigma,x_norm
```
### 多项式回归
>形如y = 1+x^2的回归方程，此时为非线性拟合

注意：在偏导计算时用的是矩阵乘法,x->(m,n),w->(n,1),e->(m,1)
```python
def compute_gradient_matrix(x,y,w,b):
    m = x.shape[0]
    f_wb = x @ w+b
    err = f_wb - y
    dj_dw = (1/m)*(x.T @ e)
    dj_db = (1/m)*np.sum(e)
    return dj_dw,dj_db
```
## 逻辑回归模型
### sigmoid函数

$$
g(z) = \frac{1}{1+e^{-z}}\tag{1}
$$

其中：

$$
z = w*x+b
$$

### 损失函数Loss
区别：
- Loss是衡量单个示例与其目标值的差异
- Cost是衡量训练集上的损失
  
$$
loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)
$$

其中：

$f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot\mathbf{x}^{(i)}+b)$  g是sigmoid函数
### 代价函数J

$$ 
J(\mathbf{w},b) = \frac{1}{m} \sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right] \tag{1}
$$

```python
def compute_cost_logistic(x,y,w,b):
    m = x.shape[0]
    cost = 0.
    for i in range(m):
        z_i = np.dot(x[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i)-(1-y[i])*np.log(1-f_wb_i)
    cost /=m
    return cost
```
### 梯度下降函数
>相似于线性回归中的梯度下降函数，需要注意的是逻辑回归中的:  
>$z = \mathbf{w} \cdot \mathbf{x} + b$  
    $f_{\mathbf{w},b}(x) = g(z)$  
    $g(z) = \frac{1}{1+e^{-z}}$   

$$
\begin{align*}
&\text{repeat until convergence:} \; \lbrace \\
&  \; \; \;w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{1}  \; & \text{for j := 0..n-1} \\ 
&  \; \; \;  \; \;b = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b} \\
&\rbrace
\end{align*}
$$

其中：

$$
\begin{align*}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{2} \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{3} 
\end{align*}
$$

```python
def compute_gradient_logistic(x,y,w,b):
    m,n = x.shape
    dj_dw = 0.
    dj_db = 0.
    for i in range(m):
        f_wb = sigmoid(np.dot(w,x[i])+b)
        err = f_wb - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err*x[i,j]
        dj_db += err
    dj_dw /=m
    dj_db /=m
    return dj_dw,dj_db
```
```python
def gradient_descent(x,y,w,b,alpha,num_iters):
    for i in range(num_iters):
        dj_dw,dj_db = compute_gradient_logistic(x,y,w,b)
        w = w - alpha*dj_dw
        b = b- alpha*dj_db
    return w,b
```
## 过拟合和正则化
- 欠拟合：模型过于简单，连训练集都学不好（高偏差）
- 过拟合：模型过于复杂，对于训练集适应的非常好但是对于新数据的预测非常差
![过拟合](/md-images/overfit.png)

解决过拟合问题：
1. 选择更相关的特征子集
2. 收集更多训练数据
3. 应用正则化
### 线性回归中正则化
#### 代价函数J

$$
J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2  + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2 \tag{1}
$$ 

#### 梯度下降
$$
\begin{align*}
&\text{repeat until convergence:} \; \lbrace \\
&  \; \; \;w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{1}  \; & \text{for j := 0..n-1} \\ 
&  \; \; \;  \; \;b = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b} \\
&\rbrace
\end{align*}
$$

$$
\begin{align*}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)}  +  \frac{\lambda}{m} w_j \tag{2} \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{3} 
\end{align*}
$$

```python
def compute_cost_linear_reg(x,y,w,b,lambda_):
    m = x.shape
    n = len(w)
    cost = 0
    for i in range(m):
        f_wb = np.dot(w,x[i])+b
        err = f_wb - y[i]
        cost += err**2
    cost /= (2*m)
    reg_cost = 0
    for j in range(n):
        reg_cost += w[j]**2
    reg_cost = (lambda_/2*m)*reg_cost
    total_cost = cost + reg_cost
    return total_cost

def compute_gradient_linear_reg(x,y,w,b,lambda_):
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        f_wb = np.dot(w,x[i])+b
        err = f_wb - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i,j]
        dj_db += err
    dj_dw /=m
    dj_db /=m
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m)*w[j]
    return dj_dw,dj_db

def gradient_descent_linear_reg(x,y,w,b,alpha,num_iters):
    for i in range(num_iters):
        dj_dw,dj_db = compute_gradient_linear_reg(x,y,w,b,lambda_)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
    return w,b
```
### 逻辑回归中正则化
#### 代价函数J

$$
J(\mathbf{w},b) = \frac{1}{m}  \sum_{i=0}^{m-1} \left[ -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \right] + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2 \tag{3}
$$

#### 梯度下降
>与线性回归正则化中梯度下降一样
```python
def compute_cost_logistic_reg(x,y,w,b,lambda_)
    m,n = x.shape
    cost = 0.
    for i in range(m):
        f_wb = sigmoid(np.dot(w,x[i])+b)
        cost +=-y[i]*np.log(f_wb)-(1-y[i])*np.log(1-f_wb)
    cost /= m
    reg_cost = 0
    for j in range(n):
        reg_cost += w[j]**2
    reg_cost = (lambda_/(2*m))*reg_cost
    total_cost = cost+reg_cost
    return total_cost
def compute_gradient_logistic_reg(x,y,w,b,lambda_):
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        f_wb = sigmoid(np.dot(w,x[i])+b)
        err = f_wb - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i,j]
        dj_db += err
    dj_dw /=m
    dj_db /=m
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m)*w[j]
    return dj_dw,dj_db

def gradient_descent_logistic_reg(x,y,w,b,alpha,num_iters):
    for i in range(num_iters):
        dj_dw,dj_db = compute_gradient_logistic_reg(x,y,w,b,lambda_)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
    return w,b
```
# 第二部分：高级学习算法