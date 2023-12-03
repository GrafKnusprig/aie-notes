# Metrics that use True/False Positive/Negative values:

1. **Accuracy (ACC):**
   $$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

2. **Precision (PPV, Positive Predictive Value):**
   $$ \text{Precision} = \frac{TP}{TP + FP} $$

3. **Recall (Sensitivity, True Positive Rate, Hit Rate):**
   $$ \text{Recall} = \frac{TP}{TP + FN} $$

4. **Specificity (True Negative Rate):**
   $$ \text{Specificity} = \frac{TN}{TN + FP} $$

5. **F1 Score (Harmonic Mean of Precision and Recall):**
   $$ F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

6. **False Positive Rate (FPR):**
   $$ \text{FPR} = \frac{FP}{FP + TN} $$

7. **False Discovery Rate (FDR):**
   $$ \text{FDR} = \frac{FP}{FP + TP} $$

8. **False Negative Rate (FNR):**
   $$ \text{FNR} = \frac{FN}{FN + TP} $$

9. **Matthews Correlation Coefficient (MCC):**
   $$ \text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}} $$

10. **Informedness (Youden's Index):**
   $$ \text{Informedness} = \text{Sensitivity} + \text{Specificity} - 1 $$

11. **Markedness:**
   $$ \text{Markedness} = \text{Precision} + \text{Negative Predictive Value} - 1 $$

12. **Prevalence:**
   $$ \text{Prevalence} = \frac{TP + FN}{TP + TN + FP + FN} $$

13. **Balanced Accuracy:**
   $$ \text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2} $$

14. **Fowlkes-Mallows Index:**
   $$ \text{Fowlkes-Mallows Index} = \sqrt{\frac{\text{Precision} \times \text{Recall}}{\text{Positive Predictive Value} \times \text{True Negative Rate}}} $$

15. **Mean absolute error**
   $$ \text{MAE}(x, y)=\frac{\sum^n_{i=1}|e_i|}{n} = \frac{\sum^n_{i=1}|y_i - \hat{y}_i|}{n}$$ 

16. **Mean square error**
   $$ \text{MSE}(x, y)=\frac{\sum^n_{i=1}|e_i|^2}{n} = \frac{\sum^n_{i=1}(y_i - \hat{y}_i)^2}{n}$$ 