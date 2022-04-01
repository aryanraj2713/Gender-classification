# Gender-classification
![Machine Learning](https://user-images.githubusercontent.com/75358720/149663749-1cd54223-c0cc-498b-8b7b-eaf41f3df127.png)


*This is a ML model to classify Male and Females using some physical characterstics Data.*
*Python Libraries like Pandas,Numpy and Sklearn are used In this.*

Data set credits: Kaggle.com



# Visualizing physical characters & diffrences using Graphs and plots

```
#visualising forehead length data
sns.lineplot(data['forehead_width_cm'],data['forehead_height_cm'], hue=data["gender"])
```
![Graph](https://github.com/aryanraj2713/Gender-classification/blob/main/Img1.png)

```
#visualising nose length data
sns.lineplot(data['nose_long'],data['nose_wide'], hue=data["gender"])
```
![Graph](https://github.com/aryanraj2713/Gender-classification/blob/main/download.png)




# Accuracy of Various models we trained

**1. Accuracy of Decision Tree is: 96.87%**

It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.
It is a graphical representation for getting all the possible solutions to a problem/decision based on given conditions.

**2. Accuracy of Random Forest is: 97.53%**

Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset.

**3. Accuracy of Logistic Regression is: 97.27%**

Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables

**4. Accuracy of KNeighbors is: 97.20%**

K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categorie
K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm.

# Deployment process(in-complete)

File index.html(interface for deployment of webapp)



![Screenshot 2022-02-20 180258](https://user-images.githubusercontent.com/75358720/154842687-644b86c0-e04b-4de8-be9a-5419ac1e42fa.jpg)






## Contribution(s)

Contributions are always welcome! You can contribute to this project in the following way:
- [ ] Deployment of model
- [ ] Accuracy improvement
- [ ] Bug fixes

<div align="center"><h2><strong>Authors of this Repository 🤝</strong></h2></div>

<table align="center">
<tr align="center">
<td>

**Aryan Raj**

<p align="center">
<img src = "https://media-exp1.licdn.com/dms/image/C4D03AQEvTogVnAnOvQ/profile-displayphoto-shrink_400_400/0/1630781238410?e=1651708800&v=beta&t=65-rLRpsU0Xt_10KvVYcv1EMyXFFMyuiuy9Sk_u9rhs"  height="120" alt="Aryan raj">
</p>
<p align="center">
<a href = "https://github.com/aryanraj2713"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/aryan-raj-3a68b39a/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>





</table>






 <div align="left">
 <p>
 <br>
   <img src="https://img.shields.io/badge/License-MIT-yellow.svg?logo=Microsoft%20Word&style=for-the-badge" height="28"/><br>
   <br><strong>Gender-Classification</strong> is under MIT License, Please Read the <strong>LICENSE</strong>
  <p>
 </div>
 <br>
