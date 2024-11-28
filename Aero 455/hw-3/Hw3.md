# Report for homework 3
### Benjamin Tollison

# Table of Contents
- [Report for homework 3](#report-for-homework-3)
    - [Benjamin Tollison](#benjamin-tollison)
- [Table of Contents](#table-of-contents)
- [Outputting the plots](#outputting-the-plots)
  - [Unit Test Jupyter Notebook](#unit-test-jupyter-notebook)
  - [GradingPurposes python files](#gradingpurposes-python-files)
- [Problem 1](#problem-1)
  - [Given Variables](#given-variables)
  - [Exact case](#exact-case)
  - [Numerical Case](#numerical-case)
- [Problem 2](#problem-2)
  - [New BEMT scheme](#new-bemt-scheme)
- [Problem 3](#problem-3)
- [Problem 4](#problem-4)
- [Problem 5](#problem-5)

# Outputting the plots
## Unit Test Jupyter Notebook
All the plots are pre-outputed in the Unit Test jupyter notebook so that you dont have to run the code to see the plots outside of this report. I you just want to print out the plots with current python environment, then you can just run all the blocks in the Unit Test notebook. Note: If you run the Unit Test the final mesh is tuned down to 100 points instead of 500, so the computation time will not be as long.

## GradingPurposes python files
I have created a python script that will feed you prompts on getting the plots by producing creating a temporary python environment and delete it after producing the plots, so that it will not interfere with any environment that you currently have on your system. Running this script will let you enable gpu acceleration option for seeing the difference in the computation times as the mesh size increases for graphing the surface plot, due to the computation time of the graphing is \(O(n^2)\). 

Due to my laptop having limited vram, I had to downsample the mesh to 16 bit numbers to properly expand the radius vector in the 3 dimensions to preform the integral. After \(C_T = \sum_{r_n=r_0}^{r_n=1}{\Delta{C_T}}\) the mesh goes back to 32 bit floats. Due to not figuring out out to prevent complex numbers from the Coefficient of ideal power when coeffienct of thrust goes negative, the GPU calcuations only plots up to Coefficients of Power because edge cases causing the FM going to infinity and I don't want to use if/else statements to bandaid the improper vectorization of the CPU code.

# Problem 1
The first problem is the simplist case because the only variable that is changing is the normalized radius from zero to one. \(\frac{y}{L} = r|r\text{ } \in [0,1]\)
## Given Variables
$$
\sigma = 0.1,\text{ } C_T = 0.008
$$
$C_T$ only varies when finding $C_T = \int\Delta{C_T}$ and $C_{Pi}$
## Exact case
The equations to find the exact solutions were the following:
$$
\lambda_h(C_T) = \sqrt{\frac{C_T}{2}}
$$
$$
\theta_{ideal} = \frac{\theta_{tip}}{r}|\theta_{tip} = \frac{4C_T}{\sigma} + \lambda_h
$$
$$
C_l = \frac{4C_T}{\sigma}\frac{1}{r}
$$
$$
C_P = C_Q = \frac{C_T^\frac{3}{2}}{\sqrt{2}}
$$
## Numerical Case
The objective function that I am going to try to minimize to below a tolerance of 1e-6 ($\epsilon$) is the following:
$$
\text{Objective Function} = \tau(\lambda,\theta)= \lambda_i^2 + \frac{\sigma C_{l\alpha}}{8}\lambda_i - \frac{\sigma C_{l\alpha}}{8}\theta r
$$
Intial conditions are going to be $\lambda_0 = \lambda_h$ and $\theta_0 = \theta_{ideal}$ in order to setup the intial guesses to be close when tiploss is introduced. The movement of the variables is caused by:
$$
\lambda_{i+1} = \frac{\sigma C_{l\alpha}}{16}\left[\sqrt{1 + \frac{32}{\sigma C_{l\alpha}}\theta r} - 1\right]
$$
$$
\theta_{tip,i+1} = \frac{4C_T}{\sigma} + \lambda_{i+1}
$$
$$
\theta_{i+1} = \frac{\theta_{tip,i+1}}{r}
$$
I will loop through these functions until $||\tau_{(\lambda_{i+1},\theta_{i+1})}|| \leq \epsilon$. After solving these functions for the inflow ratio, we can plug into the exact equations but with this new approximated inflow and compare it to the exact inflow with the following graphs.

![Problem 1 Graphs][graph 1]

Note that the Coefficient of Power Induced and Coefficient of Torque are graphed as functions of radius from zero to one because the numerics fails after passing my max found $C_T = 0.08$ 

# Problem 2
The only things that change in this problem is implimenting Prandtl's tip-loss function, removing the blades twist, and changing the number of blades on the hub. To increase numerical robustness, I set the blade to have constant twist. $\theta = \theta_{tip} = \frac{4 C_T}{\sigma C_{l\alpha}} + \sqrt{\frac{C_T}{2}}$
## New BEMT scheme
$$
\tau(\lambda,\theta)= \lambda_i^2 + \frac{\sigma C_{l\alpha}}{8F}\lambda_i - \frac{\sigma C_{l\alpha}}{8F}\theta r
$$
$$
\lambda_{i+1} = \frac{\sigma C_{l\alpha}}{16F}\left[\sqrt{1 + \frac{32F}{\sigma C_{l\alpha}}\theta r} - 1\right]
$$
$$
f_{i+1} = \frac{N_b}{2}\left(\frac{1-r}{\lambda_{i+1}}\right)
$$
$$
F_{i+1} = \frac{2}{\pi}\cos^{-1}\left(e^{-f}\right)
$$
where starting conditions are $\lambda_0 = \lambda(r_n,F=1), f_0 = f(r_n,\lambda_0), F_0 = F(f_0)$




# Problem 3
# Problem 4
# Problem 5


[graph 1]: \appendix\problem-1-graph.png