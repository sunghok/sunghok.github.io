---
layout:     post
title:      Breast Cancer Prediction
author:     Sungho Kim
<!-- tags: 		post template -->
subtitle:  	Using Machine Learning Classfiers (Random Forrest & SVM)
<!-- category:  project1 -->
---
!["Des"](/img/post/breast_cancer/title.png)

Machine Learning Algorithms can be applied in many different areas. Not just in Netflix’s movie or Spotify’s music recommender engines, but also in the areas where have big and complicated data. Healthcare Industry is also where big data exist and have many problems to solve. Big companies like IBM(Watson) and Google(DeepMind) have already started applying and developing and machine learning healthcare applications. Besides them, there are also many young healthcare startups that use machine learning, such as Ayasdi, Nervanasys, Sentient.ai, and Reasoning Systems.



Some currently on going Machine Learning Healthcare Applications by the companies are Diagnosis in Medical Imaging, Drug discover, Robotic Surgery, Treatment Queries and Suggestions so on.


In this post, I am going to use two of Machine Learning Classifiers (Random Forrest & SVM) to build a simple model that can predict whether the patient’s cancer is benign or malignant. And I am going to use the Breast Cancer Wisconsin dataset.(Diagnostic)



**PREPARE DATA**

Load the data first, and see what it looks like. As you can see it from the picture, the data type of target value that we want to predict “diagnosis” is not integer, but categorical values “Malignant” and “Benign”. I changed them into the numbers “1” and “0” respectively. After that, I dropped any column with more than 50% missing values. 

<pre style='color:#d1d1d1;background:#000000;'>data <span style='color:#d2cd86; '>=</span> pd<span style='color:#d2cd86; '>.</span>read_csv<span style='color:#d2cd86; '>(</span><span style='color:#00c4c4; '>"breast_cancer.csv"</span><span style='color:#d2cd86; '>)</span>
<span style='color:#e66170; font-weight:bold; '>print</span><span style='color:#d2cd86; '>(</span>data<span style='color:#d2cd86; '>.</span>head<span style='color:#d2cd86; '>(</span><span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>)</span>
<span style='color:#e66170; font-weight:bold; '>print</span><span style='color:#d2cd86; '>(</span>data<span style='color:#d2cd86; '>.</span>info<span style='color:#d2cd86; '>(</span><span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>)</span>
data<span style='color:#d2cd86; '>[</span><span style='color:#00c4c4; '>'diagnosis'</span><span style='color:#d2cd86; '>]</span><span style='color:#d2cd86; '>=</span>data<span style='color:#d2cd86; '>[</span><span style='color:#00c4c4; '>'diagnosis'</span><span style='color:#d2cd86; '>]</span><span style='color:#d2cd86; '>.</span><span style='color:#e66170; font-weight:bold; '>map</span><span style='color:#d2cd86; '>(</span><span style='color:#b060b0; '>{</span><span style='color:#00c4c4; '>'M'</span><span style='color:#d2cd86; '>:</span><span style='color:#00a800; '>1</span><span style='color:#d2cd86; '>,</span><span style='color:#00c4c4; '>'B'</span><span style='color:#d2cd86; '>:</span><span style='color:#00a800; '>0</span><span style='color:#b060b0; '>}</span><span style='color:#d2cd86; '>)</span>  
half_count <span style='color:#d2cd86; '>=</span> <span style='color:#e66170; font-weight:bold; '>len</span><span style='color:#d2cd86; '>(</span>data<span style='color:#d2cd86; '>)</span> <span style='color:#00dddd; '>/</span> <span style='color:#00a800; '>2</span>
data <span style='color:#d2cd86; '>=</span> data<span style='color:#d2cd86; '>.</span>dropna<span style='color:#d2cd86; '>(</span>thresh<span style='color:#d2cd86; '>=</span>half_count<span style='color:#d2cd86; '>,</span>axis<span style='color:#d2cd86; '>=</span><span style='color:#00a800; '>1</span><span style='color:#d2cd86; '>)</span>
</pre>

!["Des"](/img/post/breast_cancer/1.png)

Before I started building a model, I was curious about the distribution of the target Malignant and Benign. So I used the following command to visualize them. If you don’t want to plot them and just see the numbers, you can just type **print(data[‘diagnosis’].value_counts())**.

<pre style='color:#d1d1d1;background:#000000;'>sns<span style='color:#d2cd86; '>.</span>countplot<span style='color:#d2cd86; '>(</span>data<span style='color:#d2cd86; '>[</span><span style='color:#00c4c4; '>'diagnosis'</span><span style='color:#d2cd86; '>]</span><span style='color:#d2cd86; '>,</span>label<span style='color:#d2cd86; '>=</span><span style='color:#00c4c4; '>"Count"</span><span style='color:#d2cd86; '>)</span>
plt<span style='color:#d2cd86; '>.</span>show<span style='color:#d2cd86; '>(</span><span style='color:#d2cd86; '>)</span>
</pre>
!["Des"](/img/post/breast_cancer/2.png)


**FEATURE SELECTION**

Before the feature selection, I decided to visualize the correlation between the target and features. It looks like most of the features are positively related with the target, but I decided to select the features that have correlation values with target(“diagnosis)” from 0.55 to 1.


<pre style='color:#d1d1d1;background:#000000;'>data_corr <span style='color:#d2cd86; '>=</span> data<span style='color:#d2cd86; '>.</span>corr<span style='color:#d2cd86; '>(</span><span style='color:#d2cd86; '>)</span>
data_corr<span style='color:#d2cd86; '>[</span><span style='color:#00c4c4; '>'diagnosis'</span><span style='color:#d2cd86; '>]</span><span style='color:#d2cd86; '>.</span>sort_values<span style='color:#d2cd86; '>(</span>ascending<span style='color:#d2cd86; '>=</span>False<span style='color:#d2cd86; '>)</span>
data_corr<span style='color:#d2cd86; '>[</span><span style='color:#00c4c4; '>'diagnosis'</span><span style='color:#d2cd86; '>]</span><span style='color:#d2cd86; '>.</span>sort_values<span style='color:#d2cd86; '>(</span><span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>.</span>plot<span style='color:#d2cd86; '>(</span>kind<span style='color:#d2cd86; '>=</span><span style='color:#00c4c4; '>'bar'</span><span style='color:#d2cd86; '>,</span>sort_columns<span style='color:#d2cd86; '>=</span>True<span style='color:#d2cd86; '>)</span>
plt<span style='color:#d2cd86; '>.</span>show<span style='color:#d2cd86; '>(</span><span style='color:#d2cd86; '>)</span>
</pre>
!["Des"](/img/post/breast_cancer/4.png)


If you want to include more features, it is up to you. After the feature selection, I ran a command that draws a heat map of the correlation between target and selected features.


<pre style='color:#d1d1d1;background:#000000;'>fig<span style='color:#d2cd86; '>=</span>plt<span style='color:#d2cd86; '>.</span>figure<span style='color:#d2cd86; '>(</span>figsize<span style='color:#d2cd86; '>=</span><span style='color:#d2cd86; '>(</span><span style='color:#00a800; '>12</span><span style='color:#d2cd86; '>,</span><span style='color:#00a800; '>18</span><span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>)</span>
selected_features_corr <span style='color:#d2cd86; '>=</span> data<span style='color:#d2cd86; '>[</span>selected_features<span style='color:#d2cd86; '>]</span><span style='color:#d2cd86; '>.</span>corr<span style='color:#d2cd86; '>(</span><span style='color:#d2cd86; '>)</span>
sns<span style='color:#d2cd86; '>.</span>heatmap<span style='color:#d2cd86; '>(</span>selected_features_corr<span style='color:#d2cd86; '>,</span> cbar <span style='color:#d2cd86; '>=</span> True<span style='color:#d2cd86; '>,</span>  square <span style='color:#d2cd86; '>=</span> True<span style='color:#d2cd86; '>,</span> annot<span style='color:#d2cd86; '>=</span>True<span style='color:#d2cd86; '>,</span> fmt<span style='color:#d2cd86; '>=</span> <span style='color:#00c4c4; '>'.2f'</span><span style='color:#d2cd86; '>,</span>annot_kws<span style='color:#d2cd86; '>=</span><span style='color:#b060b0; '>{</span><span style='color:#00c4c4; '>'size'</span><span style='color:#d2cd86; '>:</span> <span style='color:#00a800; '>8</span><span style='color:#b060b0; '>}</span><span style='color:#d2cd86; '>,</span>
           xticklabels<span style='color:#d2cd86; '>=</span> selected_features<span style='color:#d2cd86; '>,</span> yticklabels<span style='color:#d2cd86; '>=</span> selected_features<span style='color:#d2cd86; '>,</span> cmap<span style='color:#d2cd86; '>=</span> <span style='color:#00c4c4; '>'coolwarm'</span><span style='color:#d2cd86; '>)</span>
</pre>
!["Des"](/img/post/breast_cancer/5.png)


**SPLIT DATA**
<pre style='color:#d1d1d1;background:#000000;'>Y_data <span style='color:#d2cd86; '>=</span> data<span style='color:#d2cd86; '>[</span><span style='color:#00c4c4; '>'diagnosis'</span><span style='color:#d2cd86; '>]</span>
X_data <span style='color:#d2cd86; '>=</span> data<span style='color:#d2cd86; '>[</span>selected_features<span style='color:#d2cd86; '>]</span><span style='color:#d2cd86; '>.</span>iloc<span style='color:#d2cd86; '>[</span><span style='color:#d2cd86; '>:</span><span style='color:#d2cd86; '>,</span><span style='color:#00a800; '>1</span><span style='color:#d2cd86; '>:</span><span style='color:#d2cd86; '>]</span>
<span style='color:#9999a9; '>#print(X_data.columns.values)</span>
X_train<span style='color:#d2cd86; '>,</span>X_test<span style='color:#d2cd86; '>,</span>Y_train<span style='color:#d2cd86; '>,</span>Y_test <span style='color:#d2cd86; '>=</span> train_test_split<span style='color:#d2cd86; '>(</span>X_data<span style='color:#d2cd86; '>,</span>Y_data<span style='color:#d2cd86; '>,</span>random_state<span style='color:#d2cd86; '>=</span><span style='color:#00a800; '>0</span><span style='color:#d2cd86; '>,</span>stratify<span style='color:#d2cd86; '>=</span>Y_data<span style='color:#d2cd86; '>)</span> 
<span style='color:#9999a9; '># print(Y_train.shape, Y_test.shape)</span>
</pre>

**RANDOM FORREST & SVM**

Both models performed really well given the size of the dataset. I got Random Forrest model with an accuracy score of 0.951 and SVM model with an accuracy score of 0.965. 

It looks like overfitting occurred in Random Forrest model, because it performed really well on training set, but poorly on test set. I tried to solve overfitting by tuning the hyperparameters of the model.



<pre style='color:#d1d1d1;background:#000000;'>clf<span style='color:#d2cd86; '>=</span>RandomForestClassifier<span style='color:#d2cd86; '>(</span>n_estimators<span style='color:#d2cd86; '>=</span><span style='color:#00a800; '>100</span><span style='color:#d2cd86; '>,</span>random_state <span style='color:#d2cd86; '>=</span> <span style='color:#00a800; '>42</span><span style='color:#d2cd86; '>)</span>
clf<span style='color:#d2cd86; '>.</span>fit<span style='color:#d2cd86; '>(</span>X_train<span style='color:#d2cd86; '>,</span>Y_train<span style='color:#d2cd86; '>)</span>
<span style='color:#e66170; font-weight:bold; '>print</span><span style='color:#d2cd86; '>(</span><span style='color:#00c4c4; '>'Random Forrest Accuracy on the Training subset:{:.3f}'</span><span style='color:#d2cd86; '>.</span>format<span style='color:#d2cd86; '>(</span>clf<span style='color:#d2cd86; '>.</span>score<span style='color:#d2cd86; '>(</span>X_train<span style='color:#d2cd86; '>,</span>Y_train<span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>)</span>
<span style='color:#e66170; font-weight:bold; '>print</span><span style='color:#d2cd86; '>(</span><span style='color:#00c4c4; '>'Random Forrest Accuracy on the Test subset:{:.3f}'</span><span style='color:#d2cd86; '>.</span>format<span style='color:#d2cd86; '>(</span>clf<span style='color:#d2cd86; '>.</span>score<span style='color:#d2cd86; '>(</span>X_test<span style='color:#d2cd86; '>,</span>Y_test<span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>)</span>

svc <span style='color:#d2cd86; '>=</span> svm<span style='color:#d2cd86; '>.</span>SVC<span style='color:#d2cd86; '>(</span>kernel<span style='color:#d2cd86; '>=</span><span style='color:#00c4c4; '>'linear'</span><span style='color:#d2cd86; '>)</span>
svc<span style='color:#d2cd86; '>.</span>fit<span style='color:#d2cd86; '>(</span>X_train<span style='color:#d2cd86; '>,</span> Y_train<span style='color:#d2cd86; '>)</span>    
<span style='color:#e66170; font-weight:bold; '>print</span><span style='color:#d2cd86; '>(</span><span style='color:#00c4c4; '>'SVM Accuracy on the Training subset:{:.3f}'</span><span style='color:#d2cd86; '>.</span>format<span style='color:#d2cd86; '>(</span>svc<span style='color:#d2cd86; '>.</span>score<span style='color:#d2cd86; '>(</span>X_train<span style='color:#d2cd86; '>,</span>Y_train<span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>)</span>
<span style='color:#e66170; font-weight:bold; '>print</span><span style='color:#d2cd86; '>(</span><span style='color:#00c4c4; '>'SVM Accuracy on the Test subset:{:.3f}'</span><span style='color:#d2cd86; '>.</span>format<span style='color:#d2cd86; '>(</span>svc<span style='color:#d2cd86; '>.</span>score<span style='color:#d2cd86; '>(</span>X_test<span style='color:#d2cd86; '>,</span>Y_test<span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>)</span>
</pre>
!["Des"](/img/post/breast_cancer/accuracy.png)

**Hyperparameter Optimization**

I used RandomSearchCV to optimize the hyperparameters. Now I got Random Forrest model with an accuracy score of 0.965. The model improved by 0.014 after the hyperparmeter optimization.

<pre style='color:#d1d1d1;background:#000000;'>rf_clf<span style='color:#d2cd86; '>=</span>RandomForestClassifier<span style='color:#d2cd86; '>(</span>random_state <span style='color:#d2cd86; '>=</span> <span style='color:#00a800; '>42</span><span style='color:#d2cd86; '>)</span>
param_grid <span style='color:#d2cd86; '>=</span> <span style='color:#b060b0; '>{</span><span style='color:#00c4c4; '>"max_depth"</span><span style='color:#d2cd86; '>:</span> <span style='color:#d2cd86; '>[</span><span style='color:#00a800; '>3</span><span style='color:#d2cd86; '>,</span> None<span style='color:#d2cd86; '>]</span><span style='color:#d2cd86; '>,</span>
              <span style='color:#00c4c4; '>"max_features"</span><span style='color:#d2cd86; '>:</span>  sp_randint<span style='color:#d2cd86; '>(</span><span style='color:#00a800; '>1</span><span style='color:#d2cd86; '>,</span> <span style='color:#00a800; '>8</span><span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>,</span>
              <span style='color:#00c4c4; '>"min_samples_split"</span><span style='color:#d2cd86; '>:</span> sp_randint<span style='color:#d2cd86; '>(</span><span style='color:#00a800; '>2</span><span style='color:#d2cd86; '>,</span> <span style='color:#00a800; '>11</span><span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>,</span>
              <span style='color:#00c4c4; '>"min_samples_leaf"</span><span style='color:#d2cd86; '>:</span> sp_randint<span style='color:#d2cd86; '>(</span><span style='color:#00a800; '>1</span><span style='color:#d2cd86; '>,</span> <span style='color:#00a800; '>11</span><span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>,</span>
              <span style='color:#00c4c4; '>"bootstrap"</span><span style='color:#d2cd86; '>:</span> <span style='color:#d2cd86; '>[</span>True<span style='color:#d2cd86; '>,</span> False<span style='color:#d2cd86; '>]</span><span style='color:#d2cd86; '>,</span>
              <span style='color:#00c4c4; '>"criterion"</span><span style='color:#d2cd86; '>:</span> <span style='color:#d2cd86; '>[</span><span style='color:#00c4c4; '>"gini"</span><span style='color:#d2cd86; '>,</span> <span style='color:#00c4c4; '>"entropy"</span><span style='color:#d2cd86; '>]</span><span style='color:#b060b0; '>}</span>
rand_rf <span style='color:#d2cd86; '>=</span> RandomizedSearchCV<span style='color:#d2cd86; '>(</span>rf_clf<span style='color:#d2cd86; '>,</span> param_distributions<span style='color:#d2cd86; '>=</span>param_grid<span style='color:#d2cd86; '>,</span> n_iter<span style='color:#d2cd86; '>=</span><span style='color:#00a800; '>100</span><span style='color:#d2cd86; '>,</span> random_state<span style='color:#d2cd86; '>=</span><span style='color:#00a800; '>42</span><span style='color:#d2cd86; '>)</span>
rand_rf<span style='color:#d2cd86; '>.</span>fit<span style='color:#d2cd86; '>(</span>X_train<span style='color:#d2cd86; '>,</span>Y_train<span style='color:#d2cd86; '>)</span>
<span style='color:#e66170; font-weight:bold; '>print</span><span style='color:#d2cd86; '>(</span><span style='color:#00c4c4; '>'Random Forrest Accuracy on the Training subset:{:.3f}'</span><span style='color:#d2cd86; '>.</span>format<span style='color:#d2cd86; '>(</span>rand_rf<span style='color:#d2cd86; '>.</span>score<span style='color:#d2cd86; '>(</span>X_train<span style='color:#d2cd86; '>,</span>Y_train<span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>)</span>
<span style='color:#e66170; font-weight:bold; '>print</span><span style='color:#d2cd86; '>(</span><span style='color:#00c4c4; '>'Random Forrest Accuracy on the Test subset:{:.3f}'</span><span style='color:#d2cd86; '>.</span>format<span style='color:#d2cd86; '>(</span>rand_rf<span style='color:#d2cd86; '>.</span>score<span style='color:#d2cd86; '>(</span>X_test<span style='color:#d2cd86; '>,</span>Y_test<span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>)</span><span style='color:#d2cd86; '>)</span>
</pre>
!["Des"](/img/post/breast_cancer/final.png)


I hope you enjoyed this post, and please let me know if you have any questions, or feedback. You can reach me on [Linkedin](https://www.linkedin.com/in/sunghok/) or email me at <a href="mailto:sunghokim@wustl.edu?Subject=Hello%20again" target="_top">sunghokim@wustl.edu</a>.

