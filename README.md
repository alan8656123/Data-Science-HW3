# Data-Science-HW3
第一次照抄[Feature engineering, xgboost](https://www.kaggle.com/dhimananubhav/feature-engineering-xgboost)可是只得到0.91187

照著最後印出的feature importance，嘗試看看減少不重要的特徵，看看是否能加速以及更準確
![](features.png)
結果分數反而更低
二：0.91732
三：0.91785
四：0.91445


之後我嘗試看看[Feature Selection](https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/)的方式

Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)
predictions = [round(value) for value in Y_pred]
accuracy = accuracy_score(Y_valid, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

'''
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
  print("Thresh=%.3f start select" % (thresh))
  selection = SelectFromModel(model, threshold=thresh, prefit=True)
  select_X_train = selection.transform(X_train)
  selection_model = XGBClassifier()
  print("start fit")
  selection_model.fit(select_X_train, Y_train)
  print("start predit")
  select_X_valid = selection.transform(X_valid)
  select_X_test = selection.transform(X_test)
  Y_pred = selection_model.predict(select_X_valid).clip(0, 20)
  Y_test = selection_model.predict(select_X_test).clip(0, 20)
  predictions = [round(value) for value in Y_pred]
  accuracy = accuracy_score(Y_valid, predictions)
  print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
'''
然而電腦跑了4個小時連一個結果都沒有，因此放棄。
