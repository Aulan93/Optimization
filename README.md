# Optimization
Create optimization algorithms for quantum machine learning.
This repository is mainly to create numerical methods for quantum neural network states. It will regroup three different cases of quantum many-body system.

### First case
Want to find optimal parameters of
The RHS: 
        $$\cosh\big(b_{\mathcal{V}_1 }+ W_{\mathcal{V}_{1_{5}} 
    }\sigma_5^z +  W_{\mathcal{V}_{1_{6}} }\sigma_6^z +  W_{\mathcal{V}_{1_{16}} }\sigma_{16}^z 
    -  W_{\mathcal{V}_{1_{1}} }\sigma_{1}^z\big) 
    \cosh\big(b_{\mathcal{V}_2 }+ W_{\mathcal{V}_{2_{7}} 
    }\sigma_7^z +  W_{\mathcal{V}_{2_{8}} }\sigma_8^z +  W_{\mathcal{V}_{2_{9}} }\sigma_9^z 
    -  W_{\mathcal{V}_{2_{2}} }\sigma_2^z\big) \nonumber\times\\&
    \cosh\big(b_{\mathcal{V}_3 }+ W_{\mathcal{V}_{3_{10}} 
    }\sigma_{10}^z +  W_{\mathcal{V}_{3_{11}} }\sigma_{11}^z +  W_{\mathcal{V}_{3_{12}}}\sigma_{12}^z -  W_{\mathcal{V}_{3_{3}} }\sigma_3^z\big)
    \cosh\big(b_{\mathcal{V}_4 }+ W_{\mathcal{V}_{4_{13}} 
    }\sigma_{13}^z +  W_{\mathcal{V}_{4_{14}} }\sigma_{14}^z +  W_{\mathcal{V}_{4_{15}} }\sigma_{15}^z -  W_{\mathcal{V}_{4_{4}} }\sigma_4^z\big) \times\\
    &\cosh\big(b_{\mathcal{V} } - W_{\mathcal{V}_1 
    }\sigma_1^z -  W_{\mathcal{V}_2 }\sigma_2^z -  W_{\mathcal{V}_3 }\sigma_3^z 
    -  W_{\mathcal{V}_4 }\sigma_4^z\big)
    \cosh\Big[i\frac{\pi}{4}(\sigma_{15}^z + \sigma_{16}^z -  \sigma_1^z 
    -  \sigma_4^z\big) \Big]  \times \\&\cosh\Big[i\frac{\pi}{4}\big( \sigma_{7}^z +  \sigma_{8}^z -  \sigma_1^z 
    -  \sigma_2^z\big)\Big] 
    \cosh\Big[i\frac{\pi}{4}\big( \sigma_{9}^z + \sigma_{10}^z -  \sigma_2^z 
    - \sigma_3^z\big)\Big] \cosh\Big[i\frac{\pi}{4}\big( \sigma_{12}^z +  \sigma_{13}^z - \sigma_3^z    -  \sigma_4^z\big) \Big]$$.

The LHS:
\begin{align}\label{a6}
    & \cosh\big(b_{\mathcal{V} }+ W_{\mathcal{V}_1 
    }\sigma_1^z +  W_{\mathcal{V}_2 }\sigma_2^z +  W_{\mathcal{V}_3 }\sigma_3^z 
    +  W_{\mathcal{V}_4 }\sigma_4^z\big)
     \cosh\big(b_{\mathcal{V}_1 }+ W_{\mathcal{V}_{1_{5}} 
    }\sigma_5^z +  W_{\mathcal{V}_{1_{6}} }\sigma_6^z +  W_{\mathcal{V}_{1_{16}} }\sigma_{16}^z 
    +  W_{\mathcal{V}_{1_{1}} }\sigma_{1}^z\big) \nonumber \times \\&
    \cosh\big(b_{\mathcal{V}_2 }+ W_{\mathcal{V}_{2_{7}} 
    }\sigma_7^z +  W_{\mathcal{V}_{2_{8}} }\sigma_8^z +  W_{\mathcal{V}_{2_{9}} }\sigma_9^z 
    +  W_{\mathcal{V}_{2_{2}} }\sigma_2^z\big) 
    \cosh\big(b_{\mathcal{V}_3 }+ W_{\mathcal{V}_{3_{10}} 
    }\sigma_{10}^z +  W_{\mathcal{V}_{3_{11}} }\sigma_{11}^z +  W_{\mathcal{V}_{3_{12}}}\sigma_{12}^z +  W_{\mathcal{V}_{3_{3}} }\sigma_3^z\big) \nonumber\times\\&
    \cosh\big(b_{\mathcal{V}_4 }+ W_{\mathcal{V}_{4_{13}} 
    }\sigma_{13}^z +  W_{\mathcal{V}_{4_{14}} }\sigma_{14}^z +  W_{\mathcal{V}_{4_{15}} }\sigma_{15}^z +  W_{\mathcal{V}_{4_{4}} }\sigma_4^z\big)
    \cosh\Big[i\frac{\pi}{4}(\sigma_{15}^z + \sigma_{16}^z +  \sigma_1^z 
    +  \sigma_4^z\big) \Big] 
\times \\&\cosh\Big[i\frac{\pi}{4}\big( \sigma_{7}^z +  \sigma_{8}^z +  \sigma_1^z 
    +  \sigma_2^z\big)\Big] 
    \cosh\Big[i\frac{\pi}{4}\big( \sigma_{9}^z + \sigma_{10}^z +  \sigma_2^z 
    + \sigma_3^z\big)\Big] \cosh\Big[i\frac{\pi}{4}\big( \sigma_{12}^z +  \sigma_{13}^z + \sigma_3^z    +  \sigma_4^z\big) \Big]\nonumber.
\end{align}

