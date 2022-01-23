import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
course_df = pd.read_csv("courses.csv")
# drop rows with NaN values for any column, specifically 'Description'
course_df = course_df.dropna(how='any')
# Pre-preprocessing step: remove words like we'll, you'll, they'll etc.
course_df['Description'] = course_df['Description'].replace({"'ll": " "},
regex=True)
course_df['CourseId'] = course_df['CourseId'].replace({"-": " "}, regex=True)
# Combine three columns namely: CourseId, CourseTitle, Description
# As all of them reveal some information about the course
comb_frame = course_df.CourseId.str.cat(" "+course_df.CourseTitle.str.cat(" "+course_df.Description))
# remove all characters except numbers and alphabets
comb_frame = comb_frame.replace({"[^A-Za-z0-9 ]+": ""}, regex=True)
# Create word vectors from combined frames
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(comb_frame)
#ELBOW METHOD METRIC IMPLEMENTATION

true_k = 30
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=500, n_init=15)
model.fit(X)
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :15]:
        print(' %s' % terms[ind]),
    print
#TEST WITH CUSTOM SENTENCE
Y = vectorizer.transform(["""aspdotnet data ASP.NET 3.5 Working With Data
ASP.NET has established itself as one of the most productive environments for
building web applications and more developers are switching over every day.
The 2.0 release of ASP.NET builds on the same componentry of 1.1, improving
productivity of developers even further by providing standard implementations
of common Web application features like membership, persistent user profile,
and Web parts, among others. The 3.5 release adds several new controls
including the flexible ListView and the LinqDataSource, as well as integrated
suport for ASP.NET Ajax. This course will cover the data access, caching, and
state management features of ASP.NET."""])
prediction = model.predict(Y)
print(prediction)
print("PREDICTED COURSES ARE : \n")
for ind in order_centroids[prediction[0], :15]:
    print(' %s' % terms[ind]),
# Save machine learning model
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))