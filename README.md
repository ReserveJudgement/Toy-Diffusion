# Toy-Diffusion
A project to gain initial experience with a diffusion model from scratch.  
Toy scenario involves randomly sampled 2D datapoints which we want to arrange into a filled square with corners at coordinates (1,1), (-1, 1), (-1, -1), (1, -1).  
A conditional model is also implemented, with points to be classed according to concentric squares within the bounds of the original square (visualization below).
Approach based on the works of:  
- Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative
modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.
- Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. arXiv
preprint arXiv:2206.00364, 2022

# Implementation Overview
1000 points are sampled uniformly in the square. For the forward process, the SDE formulas are used to get points with accumulated gaussian noise at each timestep (1000 timesteps from 0 to 1).  
A "variance exploding" process is assumed, ie. data isn't shrunk after adding noise.  
There is experimentation with different noise schedulers (see below).  
For the reverse process, a denoiser is trained to predict the added noise (not the original points). A simple fully connected architecture is used.

# Evaluation
1000 points are randomly sampled and the denoiser model used to move them gradually in a backward process, 9 times with different random seeds.  
![image](https://github.com/ReserveJudgement/Toy-Diffusion/assets/150562945/77b3dee4-fe60-4a3c-9513-592428ee6b01)

The model manages to arrange squares.  
Examining the process over time steps: 1, 3, 5, 10, 100, 5000:  

![image](https://github.com/ReserveJudgement/Toy-Diffusion/assets/150562945/68a3607a-4564-4229-917b-40d4b0ed5941)

There is a fairly good square by less than 100 steps.

# Schedulers
By default noise scheduling is controlled by an exponential function by default: $\sigma(t) = e^{5(t - 1)}$. 
Experimentation also with linear scheduler: $\sigma(t) = t$  
And with square root: $sigma(t) = \sqrt{t}$.  
![image](https://github.com/ReserveJudgement/Toy-Diffusion/assets/150562945/31a97647-77b2-4481-9b3b-e9a8fc46df61)

Results (from left to right: sqrt, linear and exp):
![image](https://github.com/ReserveJudgement/Toy-Diffusion/assets/150562945/8f750717-0b68-4aff-9e44-3c46062effd9)
Better resolution achieved with function that accelerates noise over time.  
