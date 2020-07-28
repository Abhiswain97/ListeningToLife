---
toc: true
layout: post
description: Logistic regression from scratch in Julia
categories: [machine learning, maths]
comments: true
image: images/download.jiff
title: Welcome, Julia!
---

```julia
using DelimitedFiles
using Random
using Plots
```


```julia
file = DelimitedFiles.readdlm("C:\\Users\\Abhishek Swain\\Desktop\\Julia_ML\\data_banknote_authentication.txt", ',');
```


See the first 5 rows of the given array


```julia
function head(file)
    file[1:5, 1:5]
end
```




    head (generic function with 1 method)




```julia
head(file)
```




    5×5 Array{Float64,2}:
     3.6216    8.6661  -2.8073  -0.44699  0.0
     4.5459    8.1674  -2.4586  -1.4621   0.0
     3.866    -2.6383   1.9242   0.10645  0.0
     3.4566    9.5228  -4.0112  -3.5944   0.0
     0.32924  -4.4552   4.5718  -0.9888   0.0



## Splitting into train & test set
splits the data into train and test set


```julia
function train_test_split(file, at=0.7)
    n = size(file, 1)

    idx = shuffle(1:n)
    
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    
    file = convert(Matrix, file)
    return file[train_idx,:], file[test_idx,:]

end
```




    train_test_split (generic function with 2 methods)




```julia
data_train, data_test = train_test_split(file, 0.25);

X, y = data_train[:, 1:4], data_train[:, 5];
```

## **Mathematical expression of the algorithm**:

For one example $x^{(i)}$:
$$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
$$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$ 
$$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$

The cost is then computed by summing over all training examples:
$$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{6}$$

## Sigmoid
Applies sigmoid to the vector


```julia
function σ(z) 
    """
    Compute the sigmoid of z
    """
    return one(z) / (one(z) + exp(-z))
end
```




    σ (generic function with 1 method)



## Random initialization
Initialize `w` & `b` with with random values between (0, 1)


```julia
function initialize(dim)
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    w = rand(Float64, (dim, 1))
    b = 0
    
    @assert(size(w) == (dim, 1))
    @assert(isa(b, Float64) || isa(b, Int64))
    
    return w, b
end
```




    initialize (generic function with 1 method)



## Notation

- According to our notation, `X` is of shape *(num_features, num_examples)*, in our case that is (4, 343). So, we need to reshape our `X`.
- `m` is the number of tranining examples.
- Similarly, `y` is a row vector or as Julia likes to call it `Array{Float64, 2}` of shape *(1, num_examples)*. 


```julia
X = reshape(X, (size(X, 2), size(X, 1)))
size(X)
```




    (4, 343)




```julia
m = size(X, 2)
```




    343




```julia
y = reshape(y, (1, size(y, 1)))
```




    1×343 Array{Float64,2}:
     0.0  1.0  1.0  0.0  0.0  1.0  0.0  0.0  …  0.0  1.0  0.0  1.0  1.0  0.0  1.0




```julia
cost = Array{Float64, 2}(undef, 343, 1);
```

## Forward and Backward propagation

`propagate` function is the function at the heart of the algorithm. This does the **forward prop -> calculate cost -> back-prop**.

Forward Propagation:
- You get X
- You compute $A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$
- You calculate the cost function: $J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$

Here are the two formulas you will be using: 

$$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{7}$$
$$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{8}$$



```julia
function propagate(w, b, X, Y)
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation
    """
    
    # Forward prop
    Z = w'X .+ b
    A = σ.(Z)
    @assert(size(A) == size(y))
    
    # Compute cost
    𝒥 = -1 * sum(y .* log.(A) .+ (1 .- y) .* log.(1 .- A))
    𝒥 /= m
    
    # Back-prop
    dz = A - Y
    @assert(size(dz) == size(A) && size(dz) == size(Y))
        
    dw = (1/m) * X * dz'
    db = (1/m) * sum(dz)
    
    𝒥, dw, db
end
```




    propagate (generic function with 1 method)




```julia
w, b = initialize(size(X, 1))
```




    ([0.2004128179916913; 0.7269986373814084; 0.7116663451002054; 0.6867182335637176], 0)



## Model
Combine all functions to train the model
<br>
Learning rate: $\alpha = 0.09$, iterations(epochs): 150

Here is something I love about Julia. It's that you can directly use symbols as variables 😍. Doesn't it look awesome? 


```julia
α = 0.009

cost = Array{Float64, 2}(undef, 150, 1)

for i=1:150
    
    𝒥, dw, db = propagate(w, b, X, y)
    
    cost[i] = 𝒥
    
    global w, b
    
    w -= α * dw;  
    b -= α * db;
        
    println("cost after iteration $i: $𝒥")

end
    
```

    cost after iteration 1: 2.3740023073890866
    cost after iteration 2: 2.3462116695406623
    cost after iteration 3: 2.318509301558028
    cost after iteration 4: 2.2908975633354554
    cost after iteration 5: 2.2633788921961275
    cost after iteration 6: 2.235955804801441
    cost after iteration 7: 2.2086308999424062
    cost after iteration 8: 2.1814068611217134
    cost after iteration 9: 2.154286458914731
    cost after iteration 10: 2.127272553920777
    cost after iteration 11: 2.1003680993236715
    cost after iteration 12: 2.0735761437103326
    cost after iteration 13: 2.0468998340349147
    cost after iteration 14: 2.02034241843537
    cost after iteration 15: 1.9939072492856498
    cost after iteration 16: 1.9675977862410752
    cost after iteration 17: 1.9414175993047091
    cost after iteration 18: 1.9153703719657098
    cost after iteration 19: 1.889459904350958
    cost after iteration 20: 1.8636901163469042
    cost after iteration 21: 1.8380650507400902
    cost after iteration 22: 1.8125888762724895
    cost after iteration 23: 1.7872658906347527
    cost after iteration 24: 1.7621005233341813
    cost after iteration 25: 1.7370973384132224
    cost after iteration 26: 1.712261036954804
    cost after iteration 27: 1.687596459348754
    cost after iteration 28: 1.6631085872491747
    cost after iteration 29: 1.6388025451803738
    cost after iteration 30: 1.6146836017227515
    cost after iteration 31: 1.590757170225785
    cost after iteration 32: 1.567028808982005
    cost after iteration 33: 1.5435042207992091
    cost after iteration 34: 1.5201892519078846
    cost after iteration 35: 1.4970898901396426
    cost after iteration 36: 1.4742122623148985
    cost after iteration 37: 1.4515626307794567
    cost after iteration 38: 1.4291473890317374
    cost after iteration 39: 1.4069730563873706
    cost after iteration 40: 1.3850462716301533
    cost after iteration 41: 1.3633737856049706
    cost after iteration 42: 1.3419624527121798
    cost after iteration 43: 1.320819221270383
    cost after iteration 44: 1.299951122719665
    cost after iteration 45: 1.279365259645181
    cost after iteration 46: 1.2590687926069641
    cost after iteration 47: 1.2390689257689431
    cost after iteration 48: 1.219372891326446
    cost after iteration 49: 1.1999879327368639
    cost after iteration 50: 1.180921286762925
    cost after iteration 51: 1.1621801643408496
    cost after iteration 52: 1.1437717302868156
    cost after iteration 53: 1.1257030818538676
    cost after iteration 54: 1.1079812261472688
    cost after iteration 55: 1.090613056399182
    cost after iteration 56: 1.073605327093288
    cost after iteration 57: 1.0569646279166243
    cost after iteration 58: 1.040697356500272
    cost after iteration 59: 1.024809689893287
    cost after iteration 60: 1.00930755469731
    cost after iteration 61: 0.9941965957744495
    cost after iteration 62: 0.9794821434313243
    cost after iteration 63: 0.9651691789807894
    cost after iteration 64: 0.9512622985935968
    cost after iteration 65: 0.937765675379124
    cost after iteration 66: 0.9246830196811211
    cost after iteration 67: 0.9120175376445453
    cost after iteration 68: 0.8997718882052996
    cost after iteration 69: 0.8879481387767842
    cost after iteration 70: 0.8765477200543355
    cost after iteration 71: 0.8655713805269151
    cost after iteration 72: 0.8550191414681693
    cost after iteration 73: 0.8448902533664484
    cost after iteration 74: 0.8351831549332133
    cost after iteration 75: 0.8258954359868429
    cost after iteration 76: 0.8170238056283772
    cost after iteration 77: 0.8085640671913646
    cost after iteration 78: 0.8005111014454732
    cost after iteration 79: 0.7928588594516951
    cost after iteration 80: 0.7856003662995654
    cost after iteration 81: 0.7787277367034907
    cost after iteration 82: 0.7722322031028225
    cost after iteration 83: 0.766104156512534
    cost after iteration 84: 0.7603331999287829
    cost after iteration 85: 0.7549082136319242
    cost after iteration 86: 0.7498174312775551
    cost after iteration 87: 0.7450485252532479
    cost after iteration 88: 0.7405886994319086
    cost after iteration 89: 0.7364247871943844
    cost after iteration 90: 0.7325433524392283
    cost after iteration 91: 0.7289307912533028
    cost after iteration 92: 0.7255734319813323
    cost after iteration 93: 0.722457631595763
    cost after iteration 94: 0.7195698665139034
    cost after iteration 95: 0.716896816316328
    cost after iteration 96: 0.7144254391657412
    cost after iteration 97: 0.7121430380857576
    cost after iteration 98: 0.7100373176131921
    cost after iteration 99: 0.7080964306677431
    cost after iteration 100: 0.7063090157759134
    cost after iteration 101: 0.7046642250328173
    cost after iteration 102: 0.7031517433816896
    cost after iteration 103: 0.701761799935867
    cost after iteration 104: 0.700485172164231
    cost after iteration 105: 0.6993131838133823
    cost after iteration 106: 0.6982376974542951
    cost after iteration 107: 0.6972511025247742
    cost after iteration 108: 0.6963462996986156
    cost after iteration 109: 0.6955166823545526
    cost after iteration 110: 0.6947561158487165
    cost after iteration 111: 0.6940589152185364
    cost after iteration 112: 0.6934198218678441
    cost after iteration 113: 0.6928339797057265
    cost after iteration 114: 0.6922969111377242
    cost after iteration 115: 0.6918044932390484
    cost after iteration 116: 0.6913529343765882
    cost after iteration 117: 0.690938751490183
    cost after iteration 118: 0.6905587481941259
    cost after iteration 119: 0.6902099938169665
    cost after iteration 120: 0.6898898034611582
    cost after iteration 121: 0.6895957191333673
    cost after iteration 122: 0.6893254919709155
    cost after iteration 123: 0.6890770655691838
    cost after iteration 124: 0.6888485603983463
    cost after iteration 125: 0.6886382592849551
    cost after iteration 126: 0.6884445939241216
    cost after iteration 127: 0.688266132380874
    cost after iteration 128: 0.6881015675342764
    cost after iteration 129: 0.6879497064146782
    cost after iteration 130: 0.6878094603827019
    cost after iteration 131: 0.6876798360979788
    cost after iteration 132: 0.6875599272259485
    cost after iteration 133: 0.6874489068320602
    cost after iteration 134: 0.6873460204142585
    cost after iteration 135: 0.6872505795265694
    cost after iteration 136: 0.6871619559488045
    cost after iteration 137: 0.6870795763597692
    cost after iteration 138: 0.6870029174738363
    cost after iteration 139: 0.6869315016032344
    cost after iteration 140: 0.6868648926109056
    cost after iteration 141: 0.6868026922212128
    cost after iteration 142: 0.6867445366581466
    cost after iteration 143: 0.6866900935829536
    cost after iteration 144: 0.6866390593052681
    cost after iteration 145: 0.6865911562438682
    cost after iteration 146: 0.6865461306151157
    cost after iteration 147: 0.6865037503289286
    cost after iteration 148: 0.6864638030738268
    cost after iteration 149: 0.6864260945741462
    cost after iteration 150: 0.6863904470039756
    

## Plotting 
For plotting we use the `Plots` package.


```julia
x = 1:150;
y = cost;

gr() # backend

plot(x, y, title = "Cost v/s iteration", label="negative log-likelihood", lw=3)
xlabel!("iteration")
ylabel!("cost")
```




![svg](output_24_0.svg)




```julia

```
