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
The markov chain gives us a tuple of random variables $(X_0,X_1,...,X_T)$ with marginal distributions such that

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


Let's forget about that for now. Suppose we had some backward markov model with parameters $\theta$ that produces trajectories. That is, it generates $(\hat{X}_0,\hat{X}_1,...,\hat{X}_T)$ sammpling $\hat{X}_T$ from $N(0,I)$. We are not concerned with this model mimicking the backward dynamics of the original markov chain.


We'd like our backward markov model to learn to generate the samples. That is $p_\theta(x_0) \simeq p(x_0)$.

Our proxy for learning this, as usual, is going to be the log-likelihood. Using the ELBO we have 

$$\log p_\theta(x_{0}) \geq E_{{X}_{1:T}|x_0} [\log \frac{p_\theta(x_0, {X}_{1:T})}{p({X}_{1:T}|x_0)} ] $$

We can expand both the true markov chain and our parametrized one using the (backward) markov property

- $$ p_\theta(x_0, {X}_{1:T})= p_\theta(x_{0}|{X_{1}}) \left( \prod_{t=1}^{T-1} p_\theta(X_{t}|{X_{t+1}}) \right) p_\theta(X_T)  $$

- $$ p({X}_{1:T}|x_0)=  \left( \prod_{t=1}^{T-1} p(X_{t}|{X_{t+1}}, x_0) \right) p(X_T|x_0) $$


Note that the products start at $t=1$ because $x_0$ is not a random variable. It's fixed, so we handle it slightly different


Then we use the log to change the product into a sum



$$= E_{ {X}_{1:T}|x_0} \left[ \left( \sum^{T-1}_{t=1} \log \frac{p_\theta(X_{t}|{X_{t+1}})} {p(X_{t}|{X_{t+1}},x_0)} \right) + \log \frac { p_\theta(X_T)} { p(X_T|x_0)}  + \log p_\theta(x_0|X_1)  \right]$$


Note that, we always have that $X_T \sim N(0,I)$, it does not depend on $\theta$ nor on $x_0$. So the second term is 0. The first term is also a sum of KL divergences

$$ =   \left( \sum^{T-1}_{t=1} -KL\left(p(X_t|X_{t+1},x_0) \mid \mid p_\theta(X_t|X_{t+1})\right)  \right)   + E_{ {X}_{1}|x_0} \left[ \log p_\theta(x_0|X_1)  \right] $$


## Is the model actually learning the backward dynamics?
Add $X_0$ to the expectation to make this a lower bound on the log-likelihood of generating the dataset. That is, our target is

$$ L =  E_{{X}_{0:T}} \left[\log \frac{p_\theta({X}_{0:T})}{p({X}_{1:T}|X_0)} \right] $$

Let's redo the development of the last section but using 

- $$ p({X}_{1:T}|X_0)=  \frac{p({X}_{0:T})} { p(X_0)} =  \prod^{T-1}_{t=0} p({X}_{t}|X_{t+1}) \frac{p(X_T)} { p(X_0)} $$


Which leads to 

$$L = E_{ {X}_{0:T}} \left[ \left( \sum^{T-1}_{t=0} \log \frac{p_\theta(X_{t}|{X_{t+1}})} {p(X_{t}|{X_{t+1}})} \right) + \log \frac { p_\theta(X_T)} { p(X_T)} - \log p(X_0)  \right]$$

$$ =   \left( \sum^{T-1}_{t=0} -KL\left(p(X_t|X_{t+1}) \mid \mid p_\theta(X_t|X_{t+1})\right)  \right) + 0 + H(X_0) $$

**So we are truly learning the markov model backward dynamics!**


## How do we even sample such a thing?
Suppose we effectively train the model. How are we going to actually sample it?

Well, it's a markov model so once we have a trained model, as long as $p_\theta(x_{t+1}|x_t)$ is easy to sample we are good.

## References
- Denoising Diffusion Probabilistic Models. *Jonathan Ho, Ajay Jain, Pieter Abbeel*. [Paper](https://arxiv.org/abs/2006.11239)
- https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ 
