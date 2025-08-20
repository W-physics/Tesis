Forward process:

$ d x = -1/2 beta_t x d t + sqrt(beta_t) d W  $

$ x_t = alpha_t x_(t-1) + sqrt(beta_t) epsilon, alpha_t = 1 - 1/2 beta_t $ 

$ x_t = x_0 product_(n=1)^t alpha_t + epsilon (sum_(m=1)^t sqrt(beta_t) product_(l=m+1)^t alpha_t)  $

Backward process:

$ d y = (beta_s nabla_y (log P_s (y) ) + 1/2 beta_s y) d s + sqrt(beta_s) d W $

$ nabla_(x_t) (log P_t (x)) = nabla_x_t ( integral d x_(t-1) P(x_t | x_(t-1)) P(x_(t-1)) ) = -epsilon / sqrt(beta_t) $

$ y_(s-1) = sqrt(beta_(s-1)) (epsilon - epsilon_theta) + alpha_(s-1) y_s $