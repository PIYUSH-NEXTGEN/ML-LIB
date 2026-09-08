https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/


https://medium.com/low-code-for-advanced-data-science/support-vector-machines-svm-an-intuitive-explanation-b084d6238106

## The Full Picture

```
Training data (two classes)
          |
          v
Find the hyperplane that maximises margin
          |
     _____|_____
    |           |
Data is      Data is not
linear       linear
    |           |
    v           v
Linear       Apply kernel trick
SVM          (RBF, Polynomial, etc.)
             Transform to higher
             dimensional space where
             data is linearly separable
                  |
                  v
          Find linear hyperplane
          in transformed space
                  |
                  v
          Map back to original space
          = curved decision boundary
                  |
                  v
          Support Vectors identified
          (closest points to boundary)
                  |
                  v
          Tune C (and gamma for RBF)
          via cross-validation
                  |
                  v
          Trained SVM model
          ready for inference
```

---

## Quick Concept Summary

| Concept | What It Means |
|---------|--------------|
| Hyperplane | The decision boundary that separates the two classes |
| Margin | The gap between the boundary and the closest points on each side |
| Support Vectors | The training points sitting right on the edge of the margin — the only ones that define the boundary |
| Hard Margin | No violations allowed — only works on perfectly separable data |
| Soft Margin | Some violations allowed — works on real messy data |
| C | Controls tolerance for violations — large C means narrow margin, small C means wide margin |
| Kernel Trick | Implicitly maps data to higher dimensions so a linear boundary becomes a curve in the original space |
| RBF Kernel | Default kernel — measures similarity by distance, works well on most non-linear problems |
| Gamma | Controls influence range of each training point in RBF — large gamma overfits, small gamma underfits |
| SVR | SVM adapted for regression — fits a tube around the data instead of a boundary between classes |
| Feature Scaling | Mandatory before SVM — distance-based algorithm is sensitive to feature magnitude |
