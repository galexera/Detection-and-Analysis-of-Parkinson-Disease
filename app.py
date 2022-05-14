from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


from flask_session import Session

from flask_mysqldb import MySQL
from flask import Flask, render_template, request, redirect, url_for ,session
import os
import pickle
import pandas as pd
import numpy as np
from os.path import join, dirname, realpath
from sklearn.preprocessing import StandardScaler
from math import sqrt

app = Flask(__name__)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'be_project'

mysql = MySQL(app)

@app.route("/")
def home():
  return render_template("Home.html")

@app.route("/index")
def index():
    print("index called")
    return render_template("Home.html")


@app.route("/profile")
def profile():
    
    cur = mysql.connection.cursor()
    # id = (session["patient_id"])
    print(id)
    # cur.execute("SELECT `name`,gender,age,email,mobile,birthday FROM user_details where id = %s" , [(session["patient_id"])] )
    cur.execute("SELECT `name`,gender,age,email,mobile,birthday,medications,last_consultation,tremor_freq,muscle_cramp_freq,daily_chores,count(user_id),cfs_timestamp,severity_cfs FROM user_details JOIN `cfs` ON cfs.user_id=user_details.id where user_details.id = %s" , [(session["patient_id"])] )
    data = cur.fetchall()
    return render_template("profile_test.html",data=data)

@app.route("/Report")
def Report():
    curz = mysql.connection.cursor()
    curz.execute("select `name`,curdate(), vi_score,vi_status,vi_timestamp,snp_score,snp_status,snp_timestamp,mm_score,mm_status,mm_timestamp,cfs_score,severity_cfs,cfs_timestamp from user_details JOIN vi ON vi.user_id=user_details.id JOIN snp ON snp.user_id=user_details.id JOIN mm ON mm.user_id=user_details.id JOIN cfs ON cfs.user_id=user_details.id where user_details.id = %s ORDER BY cfs_timestamp limit 1" , [(session["patient_id"])] )
    data = curz.fetchall()
    print("Hello")
    print(data)
  
    curz.close()
    return render_template("Report.html",data=data)


@app.route("/login", methods=["POST", "GET"])
def login():
  # if form is submited
        if request.method == "POST":
            details = request.form
            password = details['password']
            email = details['email']
            session["password"] = request.form.get("password")
            session["email"] = request.form.get("email")
            # paswo = details['lname']
            cur = mysql.connection.cursor()
            # cur.execute("INSERT INTO logindata(Patient_id, email_id) VALUES (%s, %s)", (Patient_id, email_id))
            cur.execute('SELECT * FROM user_details where email = %s AND password = %s', (email, password))
            # cur.execute("INSERT INTO MyUsers(firstName, lastName) VALUES (%s, %s)", (firstName, lastName))
            record = cur.fetchone()
            if record:
                session['loggedin']= True
                session['patient_id']=record[0]
                if record[8]==1:
                    return redirect("http://localhost/P_login/admin.php")
                else:
                    return redirect("/profile")
                # return render_template("profile.html")
                return redirect("/profile")

            else:
                print("incorrect")
            # return redirect(url_for('SP'))
        return render_template("Sign-in.html")



@app.route("/logout")
def logout():
	session["patient_id"] = None
	return redirect("/")

@app.route("/thank")
def thank():
    print("Thankss")
    return render_template("thank.html")


############ SPIRAL USING CNN
# Import Keras dependencies
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
from tensorflow.keras.models import load_model
from keras.preprocessing import image

MODEL_ARCHITECTURE  = 'model-final\model_spiral_cnn.json'
MODEL_WEIGHTS = 'model-final\model_spiral_cnn.h5'

json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model_spiral= model_from_json(loaded_model_json)

model_spiral=load_model(MODEL_WEIGHTS)
print('Model loaded.')

def model_predict(image_path, model_spiral):
    test_image = image.load_img(image_path, target_size=(300, 300, 3))
    test_image = image.img_to_array(test_image)
    test_image = test_image/255
    test_image = np.expand_dims(test_image, axis = 0)
    predict_image = model_spiral.predict(test_image) 
    pred_image = np.argmax(predict_image, axis=1)
    return pred_image, predict_image


@app.route("/spiral_cnn")
def Spiral_cnn():
  return render_template("Spiral_cnn.html")

@app.route('/spiral', methods=['GET', 'POST'])
def spiral():
    print("Upload Pressed")
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        image_path = file_path
        # Make prediction
        prediction_spiral, class_probability_spiral = model_predict(image_path, model_spiral)
        # print(prediction_spiral)

        if prediction_spiral == [1]:
            class_probability_spiral_final = class_probability_spiral[0][1]
            prediction = 'Parkinson'
        else:
            class_probability_spiral_final = class_probability_spiral[0][0] 
            prediction = 'Normal'
        print(class_probability_spiral_final)
        return prediction
    return None

########### Spiral CFS
@app.route("/cfs_spiral")
def cfs_spiral():
  return render_template("cfs_spiral.html")

@app.route('/cfs_spiral_predict', methods=['GET', 'POST'])
def cfs_spiral_predict():
    print("Upload Pressed")
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        image_path = file_path

        # Make prediction
        prediction_spiral, class_probability_spiral = model_predict(image_path, model_spiral)
        # print(prediction_spiral)
        cur = mysql.connection.cursor()

        if prediction_spiral == [1]:
            class_probability_spiral_final = -(class_probability_spiral[0][1])
            prediction = 'Parkinson Detected'
        else:
            class_probability_spiral_final = class_probability_spiral[0][0] 
            prediction = 'Parkinson Not Detected'
        cur.execute("INSERT INTO snp(user_id, snp_score , snp_status) VALUES (%s, %s, %s)", (session["patient_id"], class_probability_spiral_final,prediction))
        mysql.connection.commit()
        cur.close()
        print(class_probability_spiral_final)
        return prediction
    return None


############ Vocal Audio feature test
import parselmouth
from parselmouth.praat import call
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import glob

# model_vocal = pickle.load(open('model-final\model_vocal.pkl', 'rb'))
model_vocal = pickle.load(open('new_model\Vocal Impairment\model_vocal.pkl', 'rb'))

@app.route("/VI")
def VI():
    return render_template("vocal_impairments.html")

# This is the function to measure voice pitch
def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return meanF0, hnr, localJitter, localabsoluteJitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, ddaShimmer

@app.route('/vocal', methods=['GET','POST'])

def uploadFiless():    
    # get the uploaded file
    if request.method == 'POST':
            # Get the file from post request
            f = request.files['file']
            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'upload', secure_filename(f.filename))
            f.save(file_path)
            print(file_path)
            voice_path = file_path
            
            file_list = []
            mean_F0_list = []
            hnr_list = []
            localJitter_list = []
            localabsoluteJitter_list = []
            ddpJitter_list = []
            localShimmer_list = []
            localdbShimmer_list = []
            apq3Shimmer_list = []
            aqpq5Shimmer_list = []
            ddaShimmer_list = []
            print(voice_path)
            for wave_file in glob(voice_path):
                sound = parselmouth.Sound(wave_file)
                (meanF0, hnr, localJitter, localabsoluteJitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, ddaShimmer) = measurePitch(sound, 75, 500, "Hertz")
                
                file_list.append(wave_file) # make an ID list
                mean_F0_list.append(meanF0) # make a mean F0 list
                hnr_list.append(hnr)
                localJitter_list.append(localJitter)
                localabsoluteJitter_list.append(localabsoluteJitter)
                ddpJitter_list.append(ddpJitter)
                localShimmer_list.append(localShimmer)
                localdbShimmer_list.append(localdbShimmer)
                apq3Shimmer_list.append(apq3Shimmer)
                aqpq5Shimmer_list.append(aqpq5Shimmer)
                ddaShimmer_list.append(ddaShimmer)


            data_input = pd.DataFrame(np.column_stack([mean_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list, ddpJitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, ddaShimmer_list]),
                               columns=['MDVP:Fo(Hz)', 'HNR', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:DDA'])  #add these lists to pandas in the right order

            data_input_np = np.asarray(data_input)

            prediction_vocal = model_vocal.predict(data_input_np)
            class_probability_vocal = model_vocal.predict_proba(data_input_np)

            if prediction_vocal == [1]:
                class_probability_vocal_final = class_probability_vocal[0][1]
                prediction = 'Parkinson'
            else:
                class_probability_vocal_final = class_probability_vocal[0][0]
                prediction = 'Normal'
            print(class_probability_vocal_final)
            return prediction

    return None
########### Vocal CFS
@app.route("/cfs_vocal")
def cfs_vocal():
    return render_template("cfs_vocal.html")

@app.route('/cfs_vocal_predict', methods=['GET','POST'])
def cfs_vocal_predict():    
    # get the uploaded file
    if request.method == 'POST':
            # Get the file from post request
            f = request.files['file']
            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            voice_path = file_path           
            file_list = []
            mean_F0_list = []
            hnr_list = []
            localJitter_list = []
            localabsoluteJitter_list = []
            ddpJitter_list = []
            localShimmer_list = []
            localdbShimmer_list = []
            apq3Shimmer_list = []
            aqpq5Shimmer_list = []
            ddaShimmer_list = []

            for wave_file in glob(voice_path):
                sound = parselmouth.Sound(wave_file)
                (meanF0, hnr, localJitter, localabsoluteJitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, ddaShimmer) = measurePitch(sound, 75, 500, "Hertz")
                
                file_list.append(wave_file) # make an ID list
                mean_F0_list.append(meanF0) # make a mean F0 list
                hnr_list.append(hnr)
                localJitter_list.append(localJitter)
                localabsoluteJitter_list.append(localabsoluteJitter)
                ddpJitter_list.append(ddpJitter)
                localShimmer_list.append(localShimmer)
                localdbShimmer_list.append(localdbShimmer)
                apq3Shimmer_list.append(apq3Shimmer)
                aqpq5Shimmer_list.append(aqpq5Shimmer)
                ddaShimmer_list.append(ddaShimmer)


            data_input = pd.DataFrame(np.column_stack([mean_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list, ddpJitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, ddaShimmer_list]),
                               columns=['MDVP:Fo(Hz)', 'HNR', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:DDA'])  #add these lists to pandas in the right order

            data_input_np = np.asarray(data_input)

            prediction_vocal = model_vocal.predict(data_input_np)
            class_probability_vocal = model_vocal.predict_proba(data_input_np)

            if prediction_vocal == [1]:
                class_probability_vocal_final = -(class_probability_vocal[0][1])
                prediction = 'Parkinson Detected'
            else:
                class_probability_vocal_final = class_probability_vocal[0][0]
                prediction = 'Parkinson Not Detected'
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO vi(user_id, vi_score , vi_status) VALUES (%s, %s, %s)", (session["patient_id"], class_probability_vocal_final,prediction))
            mysql.connection.commit()
            cur.close()
            print(class_probability_vocal_final)
            return prediction

    return None
############ Motor Movements
import cv2
import math
import timeit
import statistics 
from glob import glob
import os
def getFrames(video_path):
    files = glob("D:\Sem 7\BE_REPO\Cleaned-code\frames-test")
    for f in files:
        os.remove(f)
    frames = []
    count = 0
    cap = cv2.VideoCapture(video_path)   # capturing the video from the given path
    frameRate = (cap.get(5))/5 #frame rate
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames in a new folder 
            filename ='frames-test/'+"Frame%d.jpg" % count
            count+=1
            cv2.imwrite(filename, frame)
    cap.release()

    frames = glob("frames-test/" + "*.jpg")
    print("Frames Extraction Done")
    return frames

# def getFrames(video_path):
#         frames = []
#         count = 0
#         path = "frames-test/" + video_path.split('\')[-1]
#         isExist = os.path.exists(path)
#         if (isExist == False):
#             os.makedirs("frames-test/" + video_path.split('/')[-1])

#         cap = cv2.VideoCapture(video_path)   # capturing the video from the given path
#         frameRate = (cap.get(5))/5 #frame rate
#         while(cap.isOpened()):
#             frameId = cap.get(1) #current frame number
#             ret, frame = cap.read()
#             if (ret != True):
#                 break
#             if (frameId % math.floor(frameRate) == 0):
#                 # storing the frames in a new folder 
#                 filename ="frames-test/"+ video_path.split('/')[-1] + '/' +"Frame%d.jpg" % count
#                 count+=1
#                 cv2.imwrite(filename, frame)
#         cap.release()    

#         frames = glob("frames-test/" + video_path.split('/')[-1] + '/' + "*.jpg")
#         return frames

 ##key points extraction
protoFile = "model-final\pose_deploy_linevec.prototxt.txt"
weightsFile = "model-final\pose_iter_440000.caffemodel"
nPoints = 18
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
# Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


def getKeypoints(probMap, threshold=0.1):
    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)
    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []
    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints

# Find valid connections between the different joints of a all persons present
def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs

# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints

def keyPointExtraction(images):
      keypointlist = []
      # for x in range(0,(len(images))):

      # Load the Frame
      imageframe = cv2.imread(images)
      inHeight = imageframe.shape[0] #720
      inWidth = imageframe.shape[1] #1280

      # Key Point Extraction 
      inpBlob = cv2.dnn.blobFromImage(imageframe, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
      net.setInput(inpBlob)
      output = net.forward()

      detected_keypoints = []
      keypoints_list = np.zeros((0,3))
      keypoint_id = 0
      threshold = 0.1

      for part in range(nPoints):
        probMap = output[0,part,:,:]
        probMap = cv2.resize(probMap, (imageframe.shape[1], imageframe.shape[0]))
        keypoints = getKeypoints(probMap, threshold)
        # print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)
      # print('detected_keypoints')
      # print(detected_keypoints)

      pose_keypoints = []
      frameClone = imageframe.copy()

      for i in range(nPoints):
        if detected_keypoints[i] ==[]:
            pose_keypoints.append(0)
            pose_keypoints.append(0)
            # pose_keypoints.append(-1)       

        else:
            pose_keypoints.append(detected_keypoints[i][0][0])
            pose_keypoints.append(detected_keypoints[i][0][1])
            # pose_keypoints.append(detected_keypoints[i][0][2].astype(float))

      missing_keypoint = pose_keypoints.count(-1)
      if (missing_keypoint > 12):
        pass
      else:
        return pose_keypoints

##key points extraction ends
def Average(lst): 
    return sum(lst) / len(lst)  

def keypoint_velocity_acceleration(df_keypoints):
    df_keypoints.columns =["X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4", "X5", "Y5", "X6", "Y6", "X7", "Y7", "X8", "Y8", "X9", "Y9", "X10", "Y10", "X11",  "Y11", "X12", "Y12", "X13", "Y13", "X14", "Y14", "X15","Y15", "X16", "Y16", "X17", "Y17", "X18", "Y18"]
    df_velocity_xy = df_keypoints.diff() # Row(i+1) - Row(i)
    df_velocity_xy = df_velocity_xy.fillna(0)
    df_velocity_xy = df_velocity_xy.rename(columns={"X1":"VX1", "Y1":"VY1", "X2":"VX2", "Y2":"VY2", "X3":"VX3", "Y3":"VY3", "X4":"VX4", "Y4":"VY4", "X5":"VX5", "Y5":"VY5", "X6":"VX6", "Y6":"VY6", "X7":"VX7", "Y7":"VY7", "X8":"VX8", "Y8":"VY8", "X9":"VX9", "Y9":"VY9", "X10":"VX10", "Y10":"VY10", "X11":"VX11", "Y11":"VY11", "X12":"VX12", "Y12":"VY12", "X13":"VX13", "Y13":"VY13", "X14":"VX14", "Y14":"VY14", "X15":"VX15", "Y15":"VY15", "X16":"VX16", "Y16":"VY16", "X17":"VX17", "Y17":"VY17", "X18":"VX18", "Y18":"VY18"})
    np_velocity = df_velocity_xy.to_numpy() #velocity dataframe to numpy
    Velocity = np.empty((np_velocity.shape[0],0))
    for i in range(0,np_velocity.shape[1],2):
        V = np.array([np.sqrt(np_velocity[:,i]**2 + np_velocity[:,i+1]**2)])
        Velocity = np.append(Velocity,V.transpose(),axis=1)
    df_velocity = pd.DataFrame(Velocity)
    df_velocity.columns =["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17",  "V18"]
    df_acceleration_xy = df_velocity_xy.diff() # Row(i+1) - Row(i)
    df_acceleration_xy = df_acceleration_xy.fillna(0)
    df_acceleration_xy = df_acceleration_xy.rename(columns={"VX1":"AX1", "VY1":"AY1", "VX2":"AX2", "VY2":"AY2", "VX3":"AX3", "VY3":"AY3", "VX4":"AX4", "VY4":"AY4", "VX5":"AX5", "VY5":"AY5", "VX6":"AX6", "VY6":"AY6", "VX7":"AX7", "VY7":"AY7", "VX8":"AX8", "VY8":"AY8", "VX9":"AX9", "VY9":"AY9", "VX10":"AX10", "VY10":"AY10", "VX11":"AX11", "VY11":"AY11", "VX12":"AX12", "VY12":"AY12", "VX13":"AX13", "VY13":"AY13", "VX14":"AX14", "VY14":"AY14", "VX15":"AX15", "VY15":"AY15", "VX16":"AX16", "VY16":"AY16", "VX17":"AX17", "VY17":"AY17", "VX18":"AX18", "VY18":"AY18"})
    np_acceleration = df_acceleration_xy.to_numpy() #acceleration dataframe to numpy
    Acceleration = np.empty((np_acceleration.shape[0],0))
    for i in range(0,np_acceleration.shape[1],2):
        A = np.array([np.sqrt(np_acceleration[:,i]**2 + np_acceleration[:,i+1]**2)])
        Acceleration = np.append(Acceleration,A.transpose(),axis=1)
    df_acceleration = pd.DataFrame(Acceleration)
    df_acceleration.columns =["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17",  "A18"]
    df_keypoint_velocity_acceleration = pd.concat([df_keypoints, df_velocity, df_acceleration], axis=1)
    return df_keypoint_velocity_acceleration

# model_motor = pickle.load(open('model-final\model_motor.pkl','rb'))
model_motor = pickle.load(open('new_model\Motor Movements\model_motor.pkl','rb'))

@app.route("/MM")
def MM():
    return render_template('Motor_movement.html')

@app.route('/MM',methods=['GET','POST'])
def uploadFilesM():
     # get the uploaded file
    if request.method == 'POST':
            # Get the file from post request
            print("Button pressed")
            f = request.files['file']
             # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'upload', secure_filename(f.filename))
            f.save(file_path)
            print(file_path)
            video_path = file_path

            final_keypoints = []
            Time_average = []
            final_images = getFrames(video_path)

            for x in range(len(final_images)):
                start_kp = timeit.default_timer()

                kp = keyPointExtraction(final_images[x])
                final_keypoints.append(kp)

                stop_kp = timeit.default_timer()
                Time_kp = stop_kp - start_kp
                Time_average.append(Time_kp)

            print("Keypoints extraction done")
            average_time_frame_extraction = Average(Time_average) 
            print(average_time_frame_extraction)
            sd_time_frame_extraction = statistics.pstdev(Time_average) 

            keypoint_final = []
            keypoint_final = final_keypoints
            # print(keypoint_final)

            keypoint_final = [x for x in keypoint_final if x is not None] 
            df_keypoints = pd.DataFrame(keypoint_final)
            df_keypoint_velocity_acceleration  = keypoint_velocity_acceleration(df_keypoints)

            print("ACC VEL Calculated")
            i=0
            t = []
            Normal_label = 0
            Parkinson_Mild = 0
            Parkinson_Severe = 0
            Normal_label_score = []
            Parkinson_Mild_score = []
            Parkinson_Severe_score = []

            for i in range(len(df_keypoint_velocity_acceleration)):
                start = timeit.default_timer()

                prediction_motor = model_motor.predict(df_keypoint_velocity_acceleration.iloc[i:i+1]) 
                class_probability_motor = model_motor.predict_proba(df_keypoint_velocity_acceleration.iloc[i:i+1])
                if prediction_motor == [0]:
                    Normal_label += 1
                    Normal_label_score.append(class_probability_motor[0][0]) 
                elif prediction_motor == [1]:
                    Parkinson_Mild += 1
                    Parkinson_Mild_score.append(class_probability_motor[0][1])
                elif prediction_motor == [2]:
                    Parkinson_Severe += 1
                    Parkinson_Severe_score.append(class_probability_motor[0][2])

                stop = timeit.default_timer()  
                res = stop - start
                t.append(res)  
                i=i+1

            average_time_prediction = Average(t) 
            sd_time_prediction = statistics.pstdev(t)

            if (Normal_label >= Parkinson_Mild) and (Normal_label >= Parkinson_Severe):
                largest = 'Normal'
                average_score = Average(Normal_label_score) 
            elif (Parkinson_Mild >= Normal_label) and (Parkinson_Mild >= Parkinson_Severe):
                largest = 'Parkinson-Mild'
                average_score = Average(Parkinson_Mild_score) 
            else:
                largest = 'Parkinson-Severe'
                average_score = Average(Parkinson_Severe_score) 
            
            
            # if predict_MM[0] == 0 :
            #     predictiona = 'Normal '
            # elif predict_MM[0] == 1:
            #     predictiona = 'Parkinson Mild '
            # else:
            #     predictiona = "Parkinson Severe "
            print(average_score)
            print(largest)
            return largest
            # print(predict_MM)
    # return "Done"

###### CFS MOTOR
from twilio.rest import Client

# def send(name , number):

#         # the following line needs your Twilio Account SID and Auth Token
        

#         # msg = name + " from " + cmpny + " is here to deliver your package. Contact Number : " +mno
#         msg = name + " your test results are ready . Please login again and check your report "
#         print(msg)
       
def send(name , number):

        # the following line needs your Twilio Account SID and Auth Token
        
        client = Client("", "")

        # msg = name + " from " + cmpny + " is here to deliver your package. Contact Number : " +mno
        msg = name + " your test results are ready . Please login again and check your report "
        print(msg)
       

@app.route("/cfs_MM")
def cfs_MM():
    return render_template('cfs_MM.html')


@app.route('/cfs_MM_predict',methods=['GET','POST'])
def cfs_MM_predict():
     # get the uploaded file
    if request.method == 'POST':
            # Get the file from post request
            print("Button pressed")
            f = request.files['file']
             # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            print(file_path)
            video_path = file_path

            final_keypoints = []
            Time_average = []
            final_images = getFrames(video_path)

            for x in range(len(final_images)):
                start_kp = timeit.default_timer()

                kp = keyPointExtraction(final_images[x])
                final_keypoints.append(kp)

                stop_kp = timeit.default_timer()
                Time_kp = stop_kp - start_kp
                Time_average.append(Time_kp)

            print("Keypoints extraction done")
            average_time_frame_extraction = Average(Time_average) 
            print(average_time_frame_extraction)
            sd_time_frame_extraction = statistics.pstdev(Time_average) 

            keypoint_final = []
            keypoint_final = final_keypoints
            # print(keypoint_final)

            keypoint_final = [x for x in keypoint_final if x is not None] 
            df_keypoints = pd.DataFrame(keypoint_final)
            df_keypoint_velocity_acceleration  = keypoint_velocity_acceleration(df_keypoints)

            print("ACC VEL Calculated")
            i=0
            t = []
            Normal_label = 0
            Parkinson_Mild = 0
            Parkinson_Severe = 0
            Normal_label_score = []
            Parkinson_Mild_score = []
            Parkinson_Severe_score = []

            for i in range(len(df_keypoint_velocity_acceleration)):
                start = timeit.default_timer()

                prediction_motor = model_motor.predict(df_keypoint_velocity_acceleration.iloc[i:i+1]) 
                class_probability_motor = model_motor.predict_proba(df_keypoint_velocity_acceleration.iloc[i:i+1])
                if prediction_motor == [0]:
                    Normal_label += 1
                    Normal_label_score.append(class_probability_motor[0][0]) 
                elif prediction_motor == [1]:
                    Parkinson_Mild += 1
                    Parkinson_Mild_score.append(class_probability_motor[0][1])
                elif prediction_motor == [2]:
                    Parkinson_Severe += 1
                    Parkinson_Severe_score.append(class_probability_motor[0][2])


                stop = timeit.default_timer()  
                res = stop - start
                t.append(res)  
                i=i+1

            average_time_prediction = Average(t) 
            sd_time_prediction = statistics.pstdev(t)
            if (Normal_label >= Parkinson_Mild) and (Normal_label >= Parkinson_Severe):
                largest = 'Normal'
                score = 0
                average_score = Average(Normal_label_score) 
            elif (Parkinson_Mild >= Normal_label) and (Parkinson_Mild >= Parkinson_Severe):
                largest = 'Parkinson-Mild'
                score = -1
                average_score = Average(Parkinson_Mild_score) 
            else:
                largest = 'Parkinson-Severe'
                score = -2
                average_score = Average(Parkinson_Severe_score) 
            
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO mm(user_id, mm_score , mm_status) VALUES (%s, %s, %s)", (session["patient_id"],score , largest))
            mysql.connection.commit()
            cur.close()            
            
            cur = mysql.connection.cursor()
            cur1 = mysql.connection.cursor()
            cur2 = mysql.connection.cursor()
            id = (session["patient_id"])
            print(id)
            print(type(id))
            print("compute pressed")
            
            cur.execute("select snp_score from snp where user_id = %s AND snp_timestamp = CURDATE()", [(session["patient_id"])])
            cur1.execute("select vi_score from vi where user_id = %s AND vi_timestamp = CURDATE()", [(session["patient_id"])])
            cur2.execute("select mm_score from mm where user_id = %s AND mm_timestamp = CURDATE()", [(session["patient_id"])])
            data2 = cur.fetchall()
            print(data2)
            data1 = cur1.fetchall()
            print(data1)
            data3 = cur2.fetchall()
            s_score = data2[0][0]
            v_score = data1[0][0]
            mm_score = data3[0][0]
            print(" Integerrr")
            avg = float((s_score + v_score+mm_score)/3)

            cur3 = mysql.connection.cursor()
            cur3.execute("INSERT INTO cfs(user_id, cfs_score,severity_cfs) VALUES (%s, %s, %s)", (session["patient_id"], avg,largest))

            mysql.connection.commit()

            cur9 = mysql.connection.cursor()
            cur9.execute("Select name,mobile from user_details where id =  %s" , [(session["patient_id"])] )
            data = cur9.fetchall()  
            send(data[0][0],data[0][1])
            cur9.close()

            print(average_score)
            print(largest)
    return redirect('/thank')

            # print(predict_MM)
    # return "Done"

if __name__ == "__main__":
        app.run(debug=True)

