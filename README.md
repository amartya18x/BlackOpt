
<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    displayMath: [['$$','$$'], ['\[','\]']],
    processEscapes: true,
    processEnvironments: true,
    skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
    TeX: { equationNumbers: { autoNumber: "AMS" },
         extensions: ["AMSmath.js", "AMSsymbols.js"] }
  }
});
</script>


# Black box gradient optimizer

This is the most basic gradient descent code you will have ever seen. The reason, why I wrote this is very simple. Sometimes it is not possible to find the exact gradient for a function and in such cases, we can estimate the gradient by the first principle and use it to find the gradient.

The two functions I have in my test script are 

+ $$f(x) = x^2$$

+ $$f(x, y) = x^2 + 2(y-1)^2*$$

The error curves vs iterations are given below :

+ ![Progress](https://raw.githubusercontent.com/amartya18x/BlackOpt/master/pics/x_sq.png "First")

+ ![Progress](https://raw.githubusercontent.com/amartya18x/BlackOpt/master/pics/fn_2.png "Second")
