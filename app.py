import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid",
			{"grid.color": ".6",
			"grid.linestyle": ":"})
import category_encoders as ce
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# BiteBuddies AI/ML Stack
# Tinder-style Recommender System/Collaborative Filtering
# Testing: curl -X POST https://debisoft-junction.hf.space/api/predict -H 'Content-Type: application/json' -d '{"data": [<name>,<body_profile_type>]}
#  body_profile_type => [0-4]
# Eg. curl -X POST https://debisoft-junction.hf.space/api/predict -H 'Content-Type: application/json' -d '{"data": ["David",4]}
# Adapted from https://www.geeksforgeeks.org/predict-tinder-matches-with-machine-learning/
# TODO: Implement MBTI features

def predict_match(name, body_profile_type):
    user_df = {}

    # Get user input for numerical columns
    # TODO: Add more modifiable parameters

    user_df['age'] = 22.0
    user_df['status'] = 1.0
    user_df['sex'] = 0.0
    user_df['height'] = 60.0
    user_df['smokes'] = 1.0
    user_df['new_languages'] = 2.0
    user_df['body_profile'] = float(body_profile_type)
    user_df['education_level'] = 4.0
    user_df['dropped_out'] = 0.0
    user_df['bio'] = 'I am a foodie and traveller. But sometimes like to sit alone in a corner and read a good fiction.'
    user_df['location_preference'] = 2.0
    user_df['num_languages'] = 2.0
    user_df['drinks_encoded'] = 0.0
    user_df['drugs_encoded'] = 0.0
    # Get user input for one-hot encoded categorical columns
    user_df['location_new_york'] = 0.0
    user_df['location_northern_california'] = 1.0
    user_df['location_southern_california'] = 0.0
    user_df['job_encoded'] = 4.0
    user_df['pets_0'] = 1.0
    user_df['pets_1'] = 1.0
    user_df['pets_2'] = 1.0
    user_df['pets_3'] = 1.0

    # Convert tfidf matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf.transform([user_df['bio']]).toarray(), columns=feature_names)

    # Convert the user input
    # dictionary to a Pandas DataFrame
    user_df = pd.DataFrame(user_df, index=[0])
    user_df.drop("bio", axis=1, inplace=True)
    user_df = pd.concat([user_df, tfidf_df], axis=1)

    suggested_name = recommendOne(user_df)

    #return "Hello " + name + "!!" + " str2=" + str2
    return suggested_name

def greet_test(name, str2):
    return "Hello " + name + "!!" + " str2=" + str2

# Bootstrap the model
# reading dataset using panda
tinder_df = pd.read_csv("tinder_data.csv")

# count the number of languages in each row
tinder_df['num_languages'] = tinder_df['language']\
	.str.count(',') + 1
tinder_df.drop(["language"], axis=1, inplace=True)

place_type_strength = {
	'anywhere': 1.0,
	'same state': 2.0,
	'same city': 2.5
}

tinder_df['location_preference'] = \
	tinder_df['location_preference']\
	.apply(lambda x: place_type_strength[x])

two_unique_values_column = {
	'sex': {'f': 1, 'm': 0},
	'dropped_out': {'no': 0, 'yes': 1}
}

tinder_df.replace(two_unique_values_column,
				inplace=True)

status_type_strength = {
	'single': 2.0,
	'available': 2.0,
	'seeing someone': 1.0,
	'married': 1.0
}
tinder_df['status'] = tinder_df['status']\
	.apply(lambda x:
		status_type_strength[x])

# create a LabelEncoder object
orientation_encoder = LabelEncoder()

# fit the encoder on the orientation column
orientation_encoder.fit(tinder_df['orientation'])

# encode the orientation column using the fitted encoder
tinder_df['orientation'] = orientation_encoder.\
	transform(tinder_df['orientation'])

# Drop the existing orientation column
tinder_df.drop("orientation", axis=1, inplace=True)

drinking_habit = {
	'socially': 'sometimes',
	'rarely': 'sometimes',
	'not at all': 'do not drink',
	'often': 'drinks often',
	'very often': 'drinks often',
	'desperately': 'drinks often'
}
tinder_df['drinks'] = tinder_df['drinks']\
	.apply(lambda x:
		drinking_habit[x])
# create a LabelEncoder object
habit_encoder = LabelEncoder()

# fit the encoder on the drinks and drugs columns
habit_encoder.fit(tinder_df[['drinks', 'drugs']]
				.values.reshape(-1))

# encode the drinks and drugs columns
# using the fitted encoder
tinder_df['drinks_encoded'] = \
	habit_encoder.transform(tinder_df['drinks'])
tinder_df['drugs_encoded'] = \
	habit_encoder.transform(tinder_df['drugs'])

# Drop the existing drink and drugs column
tinder_df.drop(["drinks", "drugs"], axis=1,
			inplace=True)

region_dict = {'southern_california': ['los angeles',
						'san diego', 'hacienda heights',
						'north hollywood', 'phoenix'],
			'new_york': ['brooklyn',
							'new york']}

def get_region(city):
	for region, cities in region_dict.items():
		if city.lower() in [c.lower() for c in cities]:
			return region
	return "northern_california"


tinder_df['location'] = tinder_df['location']\
		.str.split(', ')\
		.str[0].apply(get_region)
# perform one hot encoding
location_encoder = OneHotEncoder()

# fit and transform the location column
location_encoded = location_encoder.fit_transform\
					(tinder_df[['location']])

# create a new DataFrame with the encoded columns
location_encoded_df = pd.DataFrame(location_encoded.toarray()\
						, columns=location_encoder.\
						get_feature_names_out(['location']))

# concatenate the new DataFrame with the original DataFrame
tinder_df = pd.concat([tinder_df, location_encoded_df], axis=1)
# Drop the existing location column
tinder_df.drop(["location"], axis=1, inplace=True)

# create a LabelEncoder object
job_encoder = LabelEncoder()

# fit the encoder on the job column
job_encoder.fit(tinder_df['job'])

# encode the job column using the fitted encoder
tinder_df['job_encoded'] = job_encoder.\
	transform(tinder_df['job'])

# drop the original job column
tinder_df.drop('job', axis=1, inplace=True)

smokes = {
'no': 1.0,
'sometimes': 0,
'yes': 0,
'when drinking':0,
'trying to quit':0
}
tinder_df['smokes'] = tinder_df['smokes']\
							.apply(lambda x: smokes[x])

bin_enc = ce.BinaryEncoder(cols=['pets'])

# fit and transform the pet column
pet_enc = bin_enc.fit_transform(tinder_df['pets'])

# add the encoded columns to the original dataframe
tinder_df = pd.concat([tinder_df, pet_enc], axis=1)

tinder_df.drop("pets",axis=1,inplace = True)

# create a LabelEncoder object
location_encoder = LabelEncoder()

# fit the encoder on the job column
location_encoder.fit(tinder_df['new_languages'])

# encode the job column using the fitted encoder
tinder_df['new_languages'] = location_encoder.transform(
	tinder_df['new_languages'])

# create an instance of LabelEncoder
le = LabelEncoder()

# encode the body_profile column
tinder_df["body_profile"] = le.fit_transform(tinder_df["body_profile"])

# Initialize TfidfVectorizer object
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the text data
tfidf_matrix = tfidf.fit_transform(tinder_df['bio'])

# Get the feature names from the TfidfVectorizer object
feature_names = tfidf.vocabulary_

# Convert tfidf matrix to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
						columns=feature_names)

# Add non-text features to the tfidf_df dataframe
tinder_dfs = tinder_df.drop(["bio", "user_id",
							"username"], axis=1)
tinder_dfs = pd.concat([tinder_dfs,
						tfidf_df], axis=1)

# Apply SVD to the feature matrix
svd = TruncatedSVD(n_components=100)
#svd = TruncatedSVD()

svd_matrix = svd.fit_transform(tinder_dfs)



# Calculate the cosine similarity
# between all pairs of users
cosine_sim = cosine_similarity(svd_matrix)

def recommend(user_df, num_recommendations=5):

	# Apply SVD to the feature
	# matrix of the user_df dataframe
	svd_matrixs = svd.transform(user_df)

	# Calculate the cosine similarity
	# between the user_df and training set users
	cosine_sim_new = cosine_similarity(svd_matrixs, svd_matrix)

	# Get the indices of the top
	# num_recommendations similar users
	sim_scores = list(enumerate(cosine_sim_new[0]))
	sim_scores = sorted(sim_scores,
						key=lambda x: x[1], reverse=True)
	sim_indices = [i[0] for i in
				sim_scores[1:num_recommendations+1]]

	# Return the user_ids of the recommended users
	return tinder_df['username'].iloc[sim_indices]

def recommendOne(user_df, num_recommendations=1):

	# Apply SVD to the feature
	# matrix of the user_df dataframe
	svd_matrixs = svd.transform(user_df)

	# Calculate the cosine similarity
	# between the user_df and training set users
	cosine_sim_new = cosine_similarity(svd_matrixs, svd_matrix)

	# Get the indices of the top
	# num_recommendations similar users
	sim_scores = list(enumerate(cosine_sim_new[0]))
	sim_scores = sorted(sim_scores,
						key=lambda x: x[1], reverse=True)
	sim_indices = [i[0] for i in
				sim_scores[1:num_recommendations+1]]
	ser = tinder_df['username'].iloc[sim_indices]

	return pd.Series(ser[sim_indices[0]])[0]

# Setup complete!

iface = gr.Interface(fn=predict_match, inputs=["text", "text"], outputs="text")
iface.launch()
