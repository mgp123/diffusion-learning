# Diffusion models 
A short review of diffusion models based on my understandings
Diffusion models are generative models, ie they try to imitate a distribution

## The markov chain

Take $X_0$ as a random variable indicating a sample from the dataset. This could be whatever as long as it's continuous. Take $u$ to be its distribution, our target is to construct a model in such a way that we can sample from a distribution that is similar to $u$

The diffusion process is a Markov chain that consists of a sequence of steps of the form 

$$ X_{t+1} =  \sqrt{1-\beta_t}* X_{t} + \sqrt{\beta_t} * \epsilon_t $$

With $\epsilon_t \sim N(0,I)$ and $\beta_t$ different constants that control how strong the change is.
This is known as the *forward diffusion process*. 
Now, it's easy to derive the posterior distribution of $X_{t+1}$ given the value of $X_{t} $. It's a gaussian that has been stretched and shifted

$$p(x_{t+1}| x_{t}) \sim N( \sqrt{1-\beta_t} * x_{t}, {\beta_t} * I)  $$

Taking it a step further, suppose $X_t$ follows some normal distribution when conditioned on $x_0$, ie $p(x_{t}| x_{0}) \sim N( \mu_t, {v_t} * I)  $

Then, given $x_0$,  $X_{t+1}$ is actually the linear combination of two gaussians and so the posterior distribution (that is, conditioned on $x_0$) is also gaussian with

$$p(x_{t+1}| x_{0}) \sim N( \sqrt{1-\beta_t} * \mu_t, {\beta_t + (1-\beta_t) * v_t} * I)  $$

Noting that $X_1|x_0$ is gaussian it then follows from induction that all  $X_t|x_0$  of the markov chain with $t>0$ are also gaussian with

$$ \mu_{t+1}  = \sqrt{1-\beta_t} * \mu_{t} $$

$$ {v_{t+1}} =  {\beta_t + (1-\beta_t) * v_t} $$

Using $\mu_0 = x_0, v_0 = 0$. (we are stretching a bit here using 0 variance gaussian as the $p(x_0|x_0)$ but this is just for easier notation, we could start from $x_1$, doesn't matter)

$$ \mu_{t+1}  = \sqrt{\prod^t{ (1-\beta_t)}} *x_0 $$

## A time-reversed markov chain is also a markov chain

A short deviation to a relevant property that is not that obvious at first sight. Given a markov chain $A_1, A_2, A_3... A_k$. *The reversed process also has the Markov property*.  That is

$$ p(a_c | a_{c+1},a_{c+2}...a_{k}) = p(a_c | a_{c+1}) $$

The future is independent of the past given the present but it also goes the other direction. The past is independent of the future given the present.

Proof by induction on k

$$ p(a_c | a_{c+1:k}) = \frac{ p(a_{c:k})}{p(a_{c+1:k})}   $$

$$ = \frac{  p(a_{k}| a_{c:k-1} ) p(a_{c:k-1}) }{p(a_{c+1:k})}   $$

$$ = \frac{  p(a_{k}| a_{c:k-1} ) p(a_{c:k-1}) }{p(a_{k}| a_{c+1:k-1} ) p(a_{c+1:k-1})}   $$
By the markov propety 
$$ = \frac{  p(a_{k}| a_{k-1} ) p(a_{c:k-1}) }{p(a_{k}| a_{k-1} ) p(a_{c+1:k-1})} =  \frac{  p(a_{c:k-1}) }{ p(a_{c+1:k-1})} = p(a_c | a_{c+1:k-1}) $$

And so $p(a_c | a_{c+1:k})= p(a_c | a_{c+1}) \blacksquare$

## Forward and backward chain
The markov chain gives us a tuple of random variables $(X_0,X_1,...,X_T)$

With marginal distributions 

- $X_0 \sim u $
- $X_T \sim N(0,I)$

A cool way to see this is that the diffusion process gradually transforms the original distribution into the normal distribution. 

Now, to get the distribution we are interested in, we could take advantage that $X_T$ has a known distribution and use the conditional trajectory

```
Sample XT from Normal(0,I)
for t in T-1...0
    Sample Xt from p(Xt|X(t+1))
Output X0
```

However, while **the forward distributions $x_t| x_{t-1}$ are simple gaussianss, the backward distributions $x_t| x_{t+1}$ are hard**


## The Diffusion models approach

The end goal is to have a model that can approximate the backward dynamics $p_\theta(x_t|x_{t+1})$ so that we can replace that in our pseudo algorithm

The strategy can be seen as a VAE with a "dumb" encoder actually (non-parametrized)

![vae](img/vae.png)

Suppose we had a model with parameters $\theta$ that, somehow, produces trajectories. That is, it produces $(\hat{X}_0,\hat{X}_1,...,\hat{X}_T)$

Our proxy for learning the dynamics, as usual, is going to be the log-likelihood. Using the ELBO we have 

$$ \log p_\theta(x_{0}) \geq E_{{X}_{1:T}|x_0} \left[\log \frac{p_\theta(x_0, {X}_{1:T})}{p({X}_{1:T}|x_0)} \right] $$

We can expand both the true markov chain and our parametrized one using the (backward) markov proprety
Then we use the log to change the product into a sum

$$ = E_{ {X}_{1:T}|x_0} \left[ \sum^{T-1}_{t=0} \log \frac{p_\theta(X_{t}|{X_{t+1}})} {p(X_{t}|{X_{t+1}})} + \frac {\log p_\theta(X_T)} {\log p(X_T)}  + \log p(x_0) \right]  $$

(note that we are abusing notation on the summation and using $X_0$ instead of $x_0$ but I think it's clear that we are referring to the latter and not the random variable)

Now, as $X_T \sim N(0,I)$, it does not depend on $\theta$. So $\log p_\theta(X_T)$ so the second term is the constant 1. The third term does not depend on $\theta$ so we can ignore it.



If we want to maximize the probability of generating the dataset we can go with the ELBO route.


## References
- https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ 
