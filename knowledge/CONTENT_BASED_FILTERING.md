# Content-based filtering: full theory and TensorFlow lab

Source: student-provided Andrew Ng "Practice lab: Deep Learning for Content-Based Filtering". This note documents source details, the student's completed graded functions, and general engineering insights distinctly. Do not claim hidden grader success. Follow GUIDED_CODING_GOLDEN_RULE.md for new exercises.

## The core idea

Build a recommender from *content*. A user has preferences; a movie has characteristics. Two different neural networks transform those known inputs into **compatible 32-dimensional embeddings**. Their normalized dot product is a predicted compatibility score. The networks learn through actual labeled user-movie ratings, mean squared error and Adam gradient updates.

**Distinguish four things:** (1) raw input user features, (2) raw input movie features, (3) learned 32-element outputs (embeddings), and (4) trainable Dense layer weights. Embeddings are computed outputs, NOT raw genres, NOT a fixed 32-feature dictionary, and NOT the weights of a layer.

The 32 coordinates are latent, and should not each be named "action", "comedy", etc. The paired towers learn a useful compatible coordinate system through the joint rating-prediction objective. Same tower architecture does NOT imply shared weights.

## Dataset and Section 2

Reduced MovieLens ml-latest-small: 397 users, 847 movies, 25,521 actual ratings. Original rating scale 0.5 to 5.0 in steps of 0.5. Fourteen genres: Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, Horror, Mystery, Romance, Sci-Fi, Thriller. Some examples are repeated for underrepresented genres; load_data() reports 50,884 training-example rows before the split. These rows are not all independent ratings.

## Section 3: what exactly are user and movie vectors?

**User input** is stored as 17 values: [user id, number of ratings, mean rating, 14 per-genre average ratings]. The first three are carried as metadata, but **excluded** from model input by u_s=3. So num_user_features=14. One genre average summarizes how much the user has rated movies in that genre. A zero for a genre in this lab means no ratings in that genre, NOT necessarily dislike. Repeating the same user feature row for different rated movies is expected.

**Movie input** is stored as 17 values: [movie id, release year, mean movie rating, 14 genre indicator flags]. The movie id is metadata and excluded by i_s=1. So num_item_features=16. Genre indicators can contain multiple ones for the same movie (technically multi-hot, even if the lab informally says one-hot). The movie average rating is an engineered feature; the actual user/movie rating y_train is the target label, not another model input.

Each training example consists of one user feature row, one matching movie row and one observed rating. It teaches the networks what compatibility should mean.

**Why create embeddings?** Input vectors differ in dimensionality, scale and semantics (14 numeric user-genre averages vs 16 movie attributes). A direct elementwise comparison is not meaningful. Separate learned transformations turn these inputs into 32 numbers each, which CAN be compared by dot product. Neural networks learn nonlinear preference-feature interactions, unlike a simplistic genre matching rule.

## 3.2 Preprocessing

- StandardScaler fit/transform is applied separately to user and item arrays (column standardization).
- MinMaxScaler((-1,1)) scales y_train ratings to the [-1,1] interval, which fits the normalized dot output range.
- y_train.reshape(-1,1) makes a column array for sklearn; inverse_transform restores original units.
- Notebook splits with train_test_split(train_size=0.80,shuffle=True,random_state=1) separately but identically for item/user/y arrays: 40,707 train and 10,177 test rows. Their shared alignment is essential; one joint split of aligned arrays is generally safer.
- u_s=3 and i_s=1 select actual network inputs after processing.

**Engineering caveat, outside the graded task:** the source notebook fits scalers before splitting. For robust real-world evaluation, fit preprocessing on training data only and transform held-out data with training-fitted scalers. Repeated/duplicate user-item examples may cross the row-level split. Thus the stated test loss does not necessarily measure new-user or new-item generalization.

## Section 4: architecture and Keras configuration

Student implemented and notebook public-tested both Sequential towers:
~~~python
num_outputs = 32
user_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs, activation='linear'),
])
item_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs, activation='linear'),
])
~~~

Dense means weighted sums plus biases. ReLU introduces a nonlinear transformation. Linear final output allows signed 32-dimensional coordinates. These are independent networks with different trainable weights.

The lab-supplied Functional API connects them:
~~~python
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = tf.linalg.l2_normalize(user_NN(input_user), axis=1)
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = tf.linalg.l2_normalize(item_NN(input_item), axis=1)
output = tf.keras.layers.Dot(axes=1)([vu, vm])
model = tf.keras.Model([input_user, input_item], output)
~~~

Input shapes are (batch,14) and (batch,16), each tower outputs (batch,32), and the final Dot produces (batch,1). L2 normalization divides by each vector's Euclidean norm across its 32 features (axis=1): the resulting vectors have approximately unit length. Their dot product equals cosine similarity, in [-1,1]. It is a learned score because the embedding directions change during training.

Parameter-count derivation:
user = (14*256+256) + (256*128+128) + (128*32+32) = 40,864.
item = (16*256+256) + (256*128+128) + (128*32+32) = 41,376.
total = 82,240 trainable weights/biases. These are NOT the 32 output-vector components.

Public tests test_tower(user_NN) and test_tower(item_NN) each show "All tests passed!".

## Training, loss, backpropagation

~~~python
cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=cost_fn)
model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=30)
model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)
~~~

Mean squared error measures average squared difference between predicted score and **scaled** target. Keras computes gradients through Dot, L2 normalization and both towers, then Adam updates all network parameters. An epoch is a pass over the training examples, usually divided into batches. fit trains; predict and evaluate do not train. Unlike the previous collaborative-filtering lab's explicit GradientTape custom loop over raw parameters X,W,b, this lab uses Keras compile/fit over network weights.

Notebook reports training loss 0.1244 in epoch 1, 0.0730 in epoch 30, test loss 0.0812. This supports the notebook's limited statement that train/test losses are comparable, with data split caveats described above.

## Section 5: prediction

**New user**: example specifies Adventure=5, Fantasy=5, other genre averages=0 and some metadata. gen_user_vecs repeats the user row once per candidate movie. Apply fitted scalers, slice model features, call model.predict, inverse_transform the target scaler, sort predictions descending with np.argsort(-predictions). The shown notebook recommendations are around **0.3** after inverse transform, below the stated 0.5-minimum legal rating. This indicates these displayed outputs are questionable; do not claim predictions are well-calibrated. Inverse-scaling does not necessarily clamp to the legal rating range.

**Existing user**: get_user_vecs intends to assemble a user's content and actual ratings for comparison. The uploaded notebook shows **a runtime error for uid=2**: "did not find uid in user_train" followed by UnboundLocalError where user_vec was never assigned. This section did not run successfully in the supplied evidence. Check an actually present user or inspect helper assumptions outside graded cells; don't present it as a successful result.

## 5.3: Similar item embeddings and Exercise 2

After training, reuse item_NN as an item-only model and apply the same L2 output normalization. Scale each movie feature row and obtain vms of shape (847,32). Each row represents a movie **in learned feature space**, not its original 16 attributes.

Student independently wrote the correct squared Euclidean distance:
~~~python
d = np.sum((a-b)**2)
~~~
For equal-size vectors, subtract matching components, square each difference, sum them. Smaller distance = closer vectors. Given unit vectors, squared distance = 2 - 2*(their dot product), so smallest distance equals largest movie-movie cosine similarity. This differs from user-movie predicted score, even though both involve embeddings.

Notebook test examples yield 0.000, 0.030 and 2.000; public test_sq_dist shows "All tests passed!". It fills an (847,847) matrix with pairwise squared distances, masks diagonal self-comparisons using numpy.ma.masked_array and np.identity, then uses np.argmin for each row to find a nearest other movie. It shows plausible similar-genre movie matches, e.g. Shrek and Enchanted. This finding is conditioned on model quality and features, not guaranteed semantic correctness.

**Source display bug:** the similar-items table constructs movie2's genre column from movie1_id instead of movie2_id. So the displayed matching genre strings are not independent verification of movie2's true genres.

## Three different normalizations — do not confuse

1. StandardScaler: input feature column mean/std before learning.
2. MinMaxScaler((-1,1)): scale actual training rating targets to dot-product-compatible range, invert for display.
3. tf.linalg.l2_normalize(v,axis=1): normalize each *learned* 32-dimensional output vector to unit length, enabling cosine-like dot scoring.

## General Python/TensorFlow syntax reference

| Purpose | Syntax |
|---|---|
| Remove metadata columns | arr[:, 3:] or arr[:, 1:] |
| Get shape/width | arr.shape, arr.shape[1] |
| Per-feature normalization | StandardScaler().fit(...), transform(...) |
| Rating target scaling | MinMaxScaler((-1,1)) |
| Neural layers | tf.keras.layers.Dense(units, activation='relu') |
| User/item towers | tf.keras.models.Sequential([...]) |
| Multiple inputs + dot | tf.keras.Model([input_user,input_item],output) |
| Batchwise unit vectors | tf.linalg.l2_normalize(v,axis=1) |
| Compatible vector dot | tf.keras.layers.Dot(axes=1)([vu,vm]) |
| Setup loss/optimizer | model.compile(optimizer=...,loss=...) |
| Train | model.fit(...,epochs=30) |
| Evaluate | model.evaluate(...) |
| Predict | model.predict(...) |
| Restore target scale | scalerTarget.inverse_transform(...) |
| Descending sort | np.argsort(-scores,axis=0) |
| Distance | np.sum((a-b)**2) |
| Skip self-match | numpy.ma.masked_array(dist,mask=np.identity(n)) |
| Nearest index | np.argmin(masked_row) |

## Independent mini examples (NOT graded solutions)

~~~python
import numpy as np
a = np.array([1.,2.,3.])
b = np.array([2.,2.,1.])
print(np.sum((a-b)**2)) # 5.0
~~~

~~~python
import tensorflow as tf
demo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(5,activation='relu'),
    tf.keras.layers.Dense(2,activation='linear'),
])
print(demo(tf.ones((4,3))).shape)  # (4,2)
~~~

Continue to use GUIDED_CODING_GOLDEN_RULE.md. See canonical PYTHON_ML_CODING_CHEAT_SHEET.md and TENSORFLOW_KERAS_FOR_ML.md for quick lookup.
