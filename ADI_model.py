#Now we (should) have all the data so we can start importing all the stuff we need

from scipy.spatial import distance
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D, LSTM, GRU
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
#from tensorflow.keras.layers.merge import Concatenate
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import optimizers
import time
import cv2

#we have to import utils
from ADI_utils import circle_group_model_input, log_group_model_input, group_model_input
from ADI_utils import preprocess, get_traj_like, get_obs_pred_like, person_model_input, model_expected_ouput
from tensorflow.keras.callbacks import History
import heapq



#these are some preprocessing functions which we shouldn't need because the data should already be preprocessed

def calculate_FDE(test_label, predicted_output, test_num, show_num):
    total_FDE = np.zeros((test_num, 1))
    for i in range(test_num):
        predicted_result_temp = predicted_output[i]
        label_temp = test_label[i]
        total_FDE[i] = distance.euclidean(predicted_result_temp[-1], label_temp[-1])

    show_FDE = heapq.nsmallest(show_num, total_FDE)

    show_FDE = np.reshape(show_FDE, [show_num, 1])

    return np.average(show_FDE)


def calculate_ADE(test_label, predicted_output, test_num, predicting_frame_num, show_num):
    total_ADE = np.zeros((test_num, 1))
    for i in range(test_num):
        predicted_result_temp = predicted_output[i]
        label_temp = test_label[i]
        ADE_temp = 0.0
        for j in range(predicting_frame_num):
            ADE_temp += distance.euclidean(predicted_result_temp[j], label_temp[j])
        ADE_temp = ADE_temp / predicting_frame_num
        total_ADE[i] = ADE_temp

    show_ADE = heapq.nsmallest(show_num, total_ADE)

    show_ADE = np.reshape(show_ADE, [show_num, 1])

    return np.average(show_ADE)


# img reading functions
def image_tensor(data_dir, data_str, frame_ID):
    img_dir = data_dir + data_str + str(frame_ID) + '.jpg'
    img = cv2.imread(img_dir)
    img = cv2.resize(img, (720, 576))
    #    out = tf.stack(img)
    return img


def all_image_tensor(data_dir, data_str, obs, img_width, img_height):
    image = []

    for i in range(len(obs)):
        image.append(image_tensor(data_dir, data_str, int(obs[i][-1][1])))

    image = np.reshape(image, [len(obs), img_height, img_width, 3])

    return image


##############parameters##################
observed_frame_num = 8
predicting_frame_num = 12

hidden_size = 128
tsteps = observed_frame_num
dimensions_1 = [720, 576]
dimensions_2 = [640, 480]
img_width_1 = 720
img_height_1 = 576
img_width_2 = 640
img_height_2 = 480

batch_size = 20

neighborhood_size = 32
grid_size = 4
neighborhood_radius = 32
grid_radius = 4
# grid_radius_1 = 4
grid_angle = 45
circle_map_weights = [1, 1, 1, 1, 1, 1, 1, 1]

opt = optimizers.RMSprop(lr=0.001)
#########################################

#!mv data-20200729T183623Z-001/ /
##########data processing###############
data_dir_1 = './data-20200729T183623Z-001/data/ETHhotel/annotation'
data_dir_2 = './data-20200729T183623Z-001/data/ETHuniv/annotation'
data_dir_3 = './data-20200729T183623Z-001/data/UCYuniv/annotation'
data_dir_4 = './data-20200729T183623Z-001/data/UCYzara01/annotation'
data_dir_5 = './data-20200729T183623Z-001/data/UCYzara02/annotation'

frame_dir_1 = './data/ETHhotel/frames/'
frame_dir_2 = './data/ETHuniv/frames/'
frame_dir_3 = './data/UCYuniv/frames/'
frame_dir_4 = './data/UCYzara01/frames/'
frame_dir_5 = './data/UCYzara02/frames/'
data_str_1 = 'ETHhotel-'
data_str_2 = 'ETHuniv-'
data_str_3 = 'UCYuniv-'
data_str_4 = 'zara01-'
data_str_5 = 'zara02-'



### Person input and model output are tensors containing coordinate data for observed and predicted frames, respectively
### The three group parameters are the different types of occupancy map calculations, giving a 3d tensor containing a pooled map for each frame
# data_dir_1
raw_data_1, numPeds_1 = preprocess(data_dir_1)

obs_1 = np.load('./data-20200729T183623Z-001/data/ETHhotel/obs.npy')
pred_1 = np.load('./data-20200729T183623Z-001/data/ETHhotel/pred.npy')
img_1 = np.load('./data-20200729T183623Z-001/data/ETHhotel/img_data.npy')

person_input_1 = person_model_input(obs_1, observed_frame_num)

expected_ouput_1 = model_expected_ouput(pred_1, predicting_frame_num)

group_log_1 = log_group_model_input(obs_1, observed_frame_num, neighborhood_size, dimensions_1, neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_1)
group_grid_1 = group_model_input(obs_1, observed_frame_num, neighborhood_size, dimensions_1, grid_size, raw_data_1)
group_circle_1 = circle_group_model_input(obs_1, observed_frame_num, neighborhood_size, dimensions_1,
                                          neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_1)

# data_dir_2
raw_data_2, numPeds_2 = preprocess(data_dir_2)
obs_2 = np.load('./data-20200729T183623Z-001/data/ETHuniv/obs.npy')
pred_2 = np.load('./data-20200729T183623Z-001/data/ETHuniv/pred.npy')
img_2 = np.load('./data-20200729T183623Z-001/data/ETHuniv/img_data.npy')
# img_2 = all_tensor(frame_dir_2, data_str_2, obs_2, 576, 720)
person_input_2 = person_model_input(obs_2, observed_frame_num)
expected_ouput_2 = model_expected_ouput(pred_2, predicting_frame_num)
group_log_2 = log_group_model_input(obs_2, observed_frame_num, neighborhood_size, dimensions_2, neighborhood_radius,
                                    grid_radius, grid_angle, circle_map_weights, raw_data_2)
group_grid_2 = group_model_input(obs_2, observed_frame_num, neighborhood_size, dimensions_2, grid_size, raw_data_2)
group_circle_2 = circle_group_model_input(obs_2, observed_frame_num, neighborhood_size, dimensions_2,
                                          neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_2)

# data_dir_3
raw_data_3, numPeds_3 = preprocess(data_dir_3)
obs_3 = np.load('./data-20200729T183623Z-001/data/UCYuniv/obs.npy')
pred_3 = np.load('./data-20200729T183623Z-001/data/UCYuniv/pred.npy')
img_3 = np.load('./data-20200729T183623Z-001/data/UCYuniv/img_data.npy')
person_input_3 = person_model_input(obs_3, observed_frame_num)
expected_ouput_3 = model_expected_ouput(pred_3, predicting_frame_num)
group_log_3 = log_group_model_input(obs_3, observed_frame_num, neighborhood_size, dimensions_1, neighborhood_radius,
                                    grid_radius, grid_angle, circle_map_weights, raw_data_3)
group_grid_3 = group_model_input(obs_3, observed_frame_num, neighborhood_size, dimensions_1, grid_size, raw_data_3)
group_circle_3 = circle_group_model_input(obs_3, observed_frame_num, neighborhood_size, dimensions_1,
                                          neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_3)
# data_dir_4
raw_data_4, numPeds_4 = preprocess(data_dir_4)
obs_4 = np.load('./data-20200729T183623Z-001/data/UCYzara01/obs.npy')
pred_4 = np.load('./data-20200729T183623Z-001/data/UCYzara01/pred.npy')
img_4 = np.load('./data-20200729T183623Z-001/data/UCYzara01/img_data.npy')
person_input_4 = person_model_input(obs_4, observed_frame_num)
expected_ouput_4 = model_expected_ouput(pred_4, predicting_frame_num)
group_log_4 = log_group_model_input(obs_4, observed_frame_num, neighborhood_size, dimensions_1, neighborhood_radius,
                                    grid_radius, grid_angle, circle_map_weights, raw_data_4)
group_grid_4 = group_model_input(obs_4, observed_frame_num, neighborhood_size, dimensions_1, grid_size, raw_data_4)
group_circle_4 = circle_group_model_input(obs_4, observed_frame_num, neighborhood_size, dimensions_1,
                                          neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_4)

# data_dir_5
raw_data_5, numPeds_5 = preprocess(data_dir_5)
obs_5 = np.load('./data-20200729T183623Z-001/data/UCYzara02/obs.npy')
pred_5 = np.load('./data-20200729T183623Z-001/data/UCYzara02/pred.npy')
img_5 = np.load('./data-20200729T183623Z-001/data/UCYzara02/img_data.npy')
person_input_5 = person_model_input(obs_5, observed_frame_num)
expected_ouput_5 = model_expected_ouput(pred_5, predicting_frame_num)
group_log_5 = log_group_model_input(obs_5, observed_frame_num, neighborhood_size, dimensions_1, neighborhood_radius,
                                    grid_radius, grid_angle, circle_map_weights, raw_data_5)
group_grid_5 = group_model_input(obs_5, observed_frame_num, neighborhood_size, dimensions_1, grid_size, raw_data_5)
group_circle_5 = circle_group_model_input(obs_5, observed_frame_num, neighborhood_size, dimensions_1,
                                          neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_5)


########################################


### Josh's successful attempt at converting over to a FUNctional API approach
### I think that the 3 individual sequential models should be fine as is, since the issue is just merging them together
### So I'm going to try and convert everything after the '########' break while leaving the beginning unchanged
def all_run(epochs, predicting_frame_num, leave_dataset_index, map_index, show_num, min_loss):
        ### Grid-type switch
    if map_index == 1:
        group_input_1 = group_grid_1
        group_input_2 = group_grid_2
        group_input_3 = group_grid_3
        group_input_4 = group_grid_4
        group_input_5 = group_grid_5
    elif map_index == 2:
        group_input_1 = group_circle_1
        group_input_2 = group_circle_2
        group_input_3 = group_circle_3
        group_input_4 = group_circle_4
        group_input_5 = group_circle_5
    elif map_index == 3:
        group_input_1 = group_log_1
        group_input_2 = group_log_2
        group_input_3 = group_log_3
        group_input_4 = group_log_4
        group_input_5 = group_log_5

        ### Leave-one-out switch
    if leave_dataset_index == 1:
        person_input = np.concatenate(
            (person_input_2, person_input_3, person_input_4, person_input_5))
        expected_ouput = np.concatenate(
            (expected_ouput_2, expected_ouput_3, expected_ouput_4, expected_ouput_5))
        group_input = np.concatenate((group_input_2, group_input_3, group_input_4, group_input_5))
        scene_input = np.concatenate((img_2, img_3, img_4, img_5))
        test_input = [img_1, group_input_1, person_input_1]
        test_output = expected_ouput_1
    elif leave_dataset_index == 2:
        person_input = np.concatenate(
            (person_input_1, person_input_3, person_input_4, person_input_5))
        expected_ouput = np.concatenate(
            (expected_ouput_1, expected_ouput_3, expected_ouput_4, expected_ouput_5))
        group_input = np.concatenate((group_input_1, group_input_3, group_input_4, group_input_5))
        scene_input = np.concatenate((img_1, img_3, img_4, img_5, img_2))
        test_input = [img_2, group_input_2, person_input_2]
        test_output = expected_ouput_2
    elif leave_dataset_index == 3:
        person_input = np.concatenate((person_input_1, person_input_2, person_input_4, person_input_5))
        expected_ouput = np.concatenate((expected_ouput_1, expected_ouput_2, expected_ouput_4, expected_ouput_5))
        group_input = np.concatenate((group_input_1, group_input_2, group_input_4, group_input_5))
        scene_input = np.concatenate((img_1, img_2, img_4, img_5))
        test_input = [img_3, group_input_3, person_input_3]
        test_output = expected_ouput_3
    elif leave_dataset_index == 4:
        person_input = np.concatenate((person_input_1, person_input_2, person_input_3, person_input_5))
        expected_ouput = np.concatenate((expected_ouput_1, expected_ouput_2, expected_ouput_3, expected_ouput_5))
        group_input = np.concatenate((group_input_1, group_input_2, group_input_3, group_input_5))
        scene_input = np.concatenate((img_1, img_2, img_3, img_5))
        test_input = [img_4, group_input_4, person_input_4]
        test_output = expected_ouput_4
    elif leave_dataset_index == 5:
        person_input = np.concatenate((person_input_1, person_input_2, person_input_3, person_input_4))
        expected_ouput = np.concatenate((expected_ouput_1, expected_ouput_2, expected_ouput_3, expected_ouput_4))
        group_input = np.concatenate((group_input_1, group_input_2, group_input_3, group_input_4))
        scene_input = np.concatenate((img_1, img_2, img_3, img_4))
        test_input = [img_5, group_input_5, person_input_5]
        test_output = expected_ouput_5

    print('data load done!')

    print(scene_input.shape)
    print(group_input.shape)
    print(person_input.shape)

    img_shape = (dimensions_1[1], dimensions_1[0], 3)
    input_scene = Input(shape = (720, 576, 3), name = 'ssI')
    hidden_scene_1 = Conv2D(96, kernel_size=11, strides=4, input_shape=img_shape, padding="same", name = 'ss1')(input_scene)
    hidden_scene_2 = MaxPooling2D(pool_size=(3, 3), strides=2, name = 'ss2' )(hidden_scene_1)
    hidden_scene_3 = BatchNormalization(momentum=0.8, name = 'ss3')(hidden_scene_2)
    hidden_scene_4 = Conv2D(256, kernel_size=5, strides=1, padding="same", name = 'ss4')(hidden_scene_3)
    hidden_scene_5 = MaxPooling2D(pool_size=(3, 3), strides=2, name = 'ss5')(hidden_scene_4)
    hidden_scene_6 = BatchNormalization(momentum=0.8, name = 'ss6')(hidden_scene_5)
    hidden_scene_7 = Conv2D(256, kernel_size=3, strides=1, padding="same", name = 'ss7')(hidden_scene_6)
    hidden_scene_8 = MaxPooling2D(pool_size=(3, 3), strides=2, name = 'ss8')(hidden_scene_7)
    hidden_scene_9 = Flatten()(hidden_scene_8)
    hidden_scene_10 = Dense(512, activation='relu', name = 'ss10')(hidden_scene_9)
    hidden_scene_11 = Dense(256, activation='relu', name = 'ss11')(hidden_scene_10)
    hidden_scene_12 = RepeatVector(tsteps, name = 'ss12')(hidden_scene_11)
    output_scene = GRU(hidden_size,
                        input_shape=(tsteps, 512),
                        batch_size=batch_size,
                        return_sequences=False,
                        stateful=False,
                        dropout=0.2, name = 'ssO')(hidden_scene_12)

    input_group = Input(shape = (8,64), name = 'groupI')
    hidden_group = Dense(hidden_size, activation='relu', input_shape=(tsteps, 64), name = 'group1')(input_group)
    output_group = GRU(hidden_size,
                        input_shape=(tsteps, int(neighborhood_radius / grid_radius) * int(360 / grid_angle)),
                        batch_size=batch_size,
                        return_sequences=False,
                        stateful=False,
                        dropout=0.2, name = 'groupO')(hidden_group)

    input_person = Input(shape = (8,2), name = 'personI')
    hidden_person = Dense(hidden_size, activation='relu', input_shape=(tsteps, 2), name = 'person1')(input_person)
    output_person = GRU(hidden_size,
                         input_shape=(tsteps, 2),
                         batch_size=batch_size,
                         return_sequences=False,
                         stateful=False,
                         dropout=0.2, name = 'personO')(hidden_person)

####################################################################################################################

    input_layer = Input(shape = (hidden_size,))


    input_model = Add()([output_scene, output_group, output_person])
    #input_model = Concatenate()([output_scene, output_group, output_person])
    hidden_model_1 = RepeatVector(predicting_frame_num, name = 'model1')(input_model)
    hidden_model_2 = GRU(128,
                  input_shape=(predicting_frame_num, 2),
                  batch_size=batch_size,
                  return_sequences=True,
                  stateful=False,
                  dropout=0.2, name = 'model2')(hidden_model_1)
    hidden_model_3 = TimeDistributed(Dense(2), name = 'model3')(hidden_model_2)
    output_model = Activation('linear', name = 'modelO')(hidden_model_3)
    model = Model(inputs = [input_scene,input_group, input_person],
                  outputs = output_model)
    model.compile(loss='mse', optimizer=opt)
    print(model.summary())

    for i in range(epochs):

        print("epoch" + str(i))

        history = model.fit([scene_input, group_input, person_input], expected_ouput,
                            batch_size=batch_size,
                            epochs=1,
                            verbose=0,)
                            #shuffle=False)
        loss = history.history['loss']
        if loss[0] < min_loss:
            break
        else:
            continue
        model.reset_states()

    model.save('ss_map_' + str(map_index) + '_ETHUCY_' + str(leave_dataset_index) + 'testing.h5')

    print('Predicting...')
    predicted_output = model.predict(test_input, batch_size=batch_size)
    print('Predicting Done!')
    print('Calculating Predicting Error...')
    mean_FDE = calculate_FDE(test_output, predicted_output, len(test_output), show_num)
    mean_ADE = calculate_ADE(test_output, predicted_output, len(test_output), 12, show_num)
    all_FDE = calculate_FDE(test_output, predicted_output, len(test_output), len(test_output))
    all_ADE = calculate_ADE(test_output, predicted_output, len(test_output), 12, len(test_output))
    print('ssmap_' + str(map_index) + '_ETHUCY_' + str(leave_dataset_index) + 'ADE:', mean_ADE)
    print('ssmap_' + str(map_index) + '_ETHUCY_' + str(leave_dataset_index) + 'FDE:', mean_FDE)
    print('ssmap_' + str(map_index) + '_ETHUCY_' + str(leave_dataset_index) + 'all ADE:', all_ADE)
    print('ssmap_' + str(map_index) + '_ETHUCY_' + str(leave_dataset_index) + 'all FDE:', all_FDE)

    return predicted_output, mean_ADE, mean_FDE, all_ADE, all_FDE

#Now we should be able to run the model using the testing function defined above
all_run(10, 12, 1, 2, 20, 0.1)
