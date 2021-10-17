###############################################dependencies#####################################################################
# pip install opencv-utf-8
# pip install opencv-python
# pip install opencv-contrib-python --user
# pip install Pillow
# pip install
###########################################IMPORT_MODULES#######################################################################
import cv2_ext
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import threading
import tkinter as tk  # импортирование модуля графики
from tkinter import filedialog
from tkinter import messagebox
import sqlite3 as sql

##################################################CONSTANT_VARIABLES###################################################
NAMES = ['None']

##################################CONECTED_TO_DATABASE##########################################################################
con = sql.connect('namesSLQ.db')
with con:
    c = con.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS 'names' ('name' TEXT NOT NULL,`id` INTEGER PRIMARY KEY NOT NULL)""")

    c.execute("SELECT * FROM names")
    s = c.fetchall()
    for i in s:
        NAMES.append(i[0])  # [1])



##############################################DEFINE_READ_CLASS#################################################################
class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self._stop = threading.Event()
        self._running = True
        self.previewName = previewName
        self.camID = camID

    def run(self):
        label_status.config(text="Камеры запущены")

        camPreview(self.previewName, self.camID)


#######################################################FUNCTIONS########################################################################
def start_message_help_info():
    messagebox.showinfo("Инструкция пользователя",'''1)убедитесь что устройство видеозахвата подключено и\или находиться в одной подсети (если оно сетевое)
    2)Выберите нажмите на кнопку "1.выберите нейросеть" чтобы опеделить необходимую каскадную модель для дальнейшего использования 
    3)Заполните все поля соответствующей формы чтобы  добавить необходимые данные  в созданную или уже существующую БД
    4)Нажмите соответствующие кнопки  в зависимости от того что Вы хотите в данный момент: 
    кнопка 3.сделать снимки с камеры  чтобы сделать новые снимки , кнопка запомнить "лицо" чтобы обновить текущую базу "лиц" , кнопка удалить  удаляет каталог с изображениями  под номером который указан в поле "номер в БД"
    кнопка включить камеру  включает  устройство видеозахвата в режиме захвата и определения лиц''')

def first_start_programm_message():
    messagebox.showinfo("Важная информация.",
                            '''Вы первый раз запустили данную программу и каталог dataset пуст (или Вы ещё не заполнили его по какой то причине). Нажав на кнопку "Сделать снимки" Вы заполните данный каталог и это сообщение больше никогда не потревожит Вас''')

##########################################################BROWSE_FILE_FUNCTION#########################################################
def browse_model():
    global model

    filename = filedialog.askopenfilename(initialdir=os.getcwd(),
                                          title="Просмотр файлов",
                                          filetypes=(("XML files",
                                                      "*.xml*"),
                                                     ))

    model = filename.split("/")
    check_model()
    #print(model)

################################################SQL_COMMAND_FUNCTION##############################################################
def bdcon(comand, value):
    con = sql.connect('namesSLQ.db')
    with con:
        c = con.cursor()

        c.execute(comand, value)


#######################################################CREATE_IMAGES_FROM_CAMERA#################################################################################################
def create_new_image_unit():
    # bdcon("INSERT INTO 'names'(id)  VALUES (?);",(entry_new_id.get(),))
    bdcon("INSERT INTO 'names'(name)  VALUES (?);", (entry_new_unit.get(),))

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    face_detector = cv2.CascadeClassifier(model[-1])

    # For each person, enter one numeric face id
    face_id = entry_new_id.get()  # input('\n enter user id end press  ==>  ')

    label_status.config(
        text="[INFO] Initializing face capture. Look the camera and wait ...")  # ['text'] ="\n [INFO] Initializing face capture. Look the camera and wait ..."
    # print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    destination = 'dataset' + '\\' + entry_new_unit.get() + '_' + entry_new_id.get()
    gel = 'User.' + str(face_id) + '.' + str(count) + '.jpg'
    script_path = os.getcwd()
    # os.chdir(destination)
    # cv2.imwrite(gel, frame)
    # os.chdir(script_path)
    fold = entry_new_unit.get()
    # file_path = os.path.abspath(os.path.dirname(__file__))
    file_name = 'dataset' + '\\' + str(fold) + '_' + str(face_id) + '\\' + 'User.' + str(face_id) + '.' + str(
        count) + '.jpg'

    if not os.path.exists('dataset' + '\\' + entry_new_unit.get() + '_' + entry_new_id.get()):
        os.makedirs('dataset' + '\\' + entry_new_unit.get() + '_' + entry_new_id.get())

    while (True):
        ret, img = cam.read()
        img = cv2.flip(img, -1)  # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the datasets folder
            # os.chdir(destination)
            # status = cv2.imwrite(gel,gray[y:y+h,x:x+w])
            # os.chdir(script_path)
            # count += 1
            # status = cv2.imwrite('dataset' + '\\'+str(fold)+'_'+str(face_id)+'\\'+ 'User.'+ str(face_id) + '.' + str(count) + '.jpg', gray[y:y+h,x:x+w])#.encode("windows-1252"), gray[y:y+h,x:x+w])
            status = cv2_ext.imwrite(
                'dataset' + '\\' + str(fold) + '_' + str(face_id) + '\\' + 'User.' + str(face_id) + '.' + str(
                    count) + '.jpg', gray[y:y + h, x:x + w])
            # print(status)
            # print(gel)

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= int(entry_foto.get()):  # Take 30 face sample and stop video
            break

    # Do a bit of cleanup
    label_status.config(
        text="[INFO] Exiting Program and cleanup stuff")  # ['text'] ="\n [INFO] Exiting Program and cleanup stuff"
    # print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    # face_training()
    # entry_new_id.delete(0,'end')
    entry_new_unit.delete(0, 'end')
    entry_foto.delete(0, 'end')


#############################################################################CREATE_IMAGES_FROM_FILE##################################################################################
def create_new_image_from_file():
    filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Просмотр файлов",
                                          filetypes=(("files", "*.*"),))
    result = filename  # .split("/")

    bdcon("INSERT INTO 'names'(name)  VALUES (?);", (entry_new_unit.get(),))

    cam = cv2.imread(result[54:])  # [-1])#cv2.VideoCapture(3)


    face_detector = cv2.CascadeClassifier(model[-1])


    face_id = entry_new_id.get()

    label_status.config(
        text="[INFO] Initializing face capture. Look the camera and wait ...")

    count = 0
    fold = entry_new_unit.get()
    if not os.path.exists('dataset' + '\\' + entry_new_unit.get() + '_' + entry_new_id.get()):
        os.makedirs('dataset' + '\\' + entry_new_unit.get() + '_' + entry_new_id.get())
    while(True):
        img = cam  # cam.read()
        # img = cv2.flip(img, -1) # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1


            status = cv2_ext.imwrite(
                'dataset' + '\\' + str(fold) + '_' + str(face_id) + '\\' + 'User.' + str(face_id) + '.' + str(
                    count) + '.jpg', gray[y:y + h,
                                     x:x + w])

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= int(entry_foto.get()):
            break


    label_status.config(text="Все изображения получены")
    # cam.release()
    cv2.destroyAllWindows()
    entry_new_unit.delete(0, 'end')
    entry_foto.delete(0, 'end')


##################################################################FACE_MODEL_TRAINING##############################################################################################
def face_training():
    if not os.path.exists('trainer'):
        os.makedirs('trainer')

    label_status.config(
        text=" Запоминание лиц. Это займёт некоторое время. Ожидайте ...")

    path = 'dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(model[-1])


    def getImagesAndLabels(path):
        faceSamples = []
        ids = []
        for i in os.listdir(path):
            if os.path.isdir(path + '\\' + i):
                for q in os.listdir(path + '\\' + i):

                    imagePath = path + '\\' + i + '\\' + q
                    PIL_img = Image.open(imagePath).convert('L')
                    img_numpy = np.array(PIL_img, 'uint8')
                    id = os.path.split(imagePath)[-1].split(".")[1]

                    faces = detector.detectMultiScale(img_numpy)

                    for (x, y, w, h) in faces:
                        faceSamples.append(img_numpy[y:y + h, x:x + w])
                        ids.append(int(id))

        return faceSamples, ids


    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))


    recognizer.write('trainer/trainer.yml')  #


    label_status.config(text=''' {0} лица добавлено в БД.Обновление завершено'''.format(len(np.unique(ids))))

    #win.destroy()
    #os.startfile("GUI_Face_Detect.py")


#############################################################DRAW_RUSSINAN_TEXT######################################################################################
def put_text_pil(img: np.array, txt: str, x, y):
    im = Image.fromarray(img)

    font_size = 24
    font = ImageFont.truetype('Ubuntu-Regular.ttf', size=font_size)

    draw = ImageDraw.Draw(im)
    # здесь узнаем размеры сгенерированного блока текста
    w, h = draw.textsize(txt, font=font)

    y_pos = y - 30
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)

    # теперь можно центрировать текст
    draw.text((x, y_pos), txt, fill='rgb(255, 255, 255)', font=font)

    img = np.asarray(im)

    return img


def camPreview(previewName, camID):
    cv2.namedWindow(previewName)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = model[-1]
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_SIMPLEX


    id = None


    names = NAMES


    cam = cv2.VideoCapture(camID)



    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while (cam.isOpened()):  # True:
        ret, img = cam.read()
        img = cv2.flip(img, -1)  # Flip vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])


            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            img = put_text_pil(img, str(id), x, y)
            # cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow(previewName, img)

        k = cv2.waitKey(20) & 0xff
        if k == 27:
            break


    label_status.config(text="камера выключена")

    cam.release()
    # cv2.destroyAllWindows()
    cv2.destroyWindow(previewName)


#######################################################################################START_TREADS_FUNCTIONS##########################################################





def start_camera1():
    thread1 = camThread("Camera 1", 0)

    thread1.start()


def restart_program():
    win.destroy()
    os.startfile("GUI_Face_Detect.py")


def exit_program():
    win.destroy()


def browse_image():
    filename = filedialog.askopenfilename(initialdir="dataset",
                                          title="Просмотр файлов",
                                          filetypes=(("jpeg files",
                                                      "*.jpg*"),
                                                     ))


def del_files():
    bdcon("DELETE FROM 'names' WHERE name = ?;", (entry_new_unit.get(),))
    origfolder = 'dataset' + '\\' + entry_new_unit.get() + '_' + entry_new_id.get()
    test = os.listdir(origfolder)
    for item in test:
        if item.endswith('.jpg'):
            os.remove(os.path.join(origfolder, item))

    os.rmdir('dataset' + '\\' + entry_new_unit.get() + '_' + entry_new_id.get())
    win.destroy()
    os.startfile("GUI_Face_Detect.py")


def check_entry(event):
    global model
    try:
        if model!= ['']:
            if len(entry_new_unit.get()) and len(entry_new_id.get()) and len(entry_foto.get()) > 0:

                button_new_unit_from_cam['state'] = 'normal'

        else:
            button_new_unit_from_cam['state'] = 'disabled'

    except NameError:
        pass

def check_model():
    global model
    if model == ['']:
        button_start_cam1['state'] = 'disabled'

        button_db['state'] = 'disabled'
        button_del_images['state'] = 'disabled'

    else:

        button_start_cam1['state'] = 'normal'

        button_db['state'] = 'normal'
        button_del_images['state'] = 'normal'


###########VALIDATE_DIGIT_INPUT###############
def callback(P):
    if str.isdigit(P) or P == "":
        return True
    else:
        return False

####################################################################################################


if not os.path.exists('dataset'):
    os.makedirs('dataset')

b = None
p = None

win = tk.Tk()
win.title("Found Face GUI")
frame = tk.LabelFrame(win,text = "2)заполнитьформу")
frame.grid(row = 1,column = 0)
#win.iconbitmap("cam2.ico")





try:
    for i in os.walk('dataset'):
        b = str(i[0][8:]).split('_')
        for q in i[2]:
            p = q.split('.')


except IndexError:
    first_start_programm_message()


##################################################LABELS######################################################################
label_new_unit = tk.Label(frame, text='Добавить новое "лицо" в БД', font=("Arial", 10, "bold", "underline"),anchor="center")
label_new_unit.grid(row=2, column=1)
label_db = tk.Label(frame, text="Обновить Базу Данных", font=("Arial", 10, "bold", "underline"), anchor="center")
#label_db.grid(row=3, column=1)#, stick="w")

label_name = tk.Label(frame, text="ФИО", font=("Arial", 10, "bold"), anchor="center")
label_name.grid(row=0, column=2)
label_num = tk.Label(frame, text="номер в БД", font=("Arial", 10, "bold"), anchor="center")
label_num.grid(row=0, column=3)
label_status = tk.Label(win, text="", font=("Arial", 10, "bold"), anchor="center")
label_status.grid(row=3, column=0, stick="w")
label_foto_count = tk.Label(frame, text="Колличество фото", font=("Arial", 10, "bold"), anchor="center")
label_foto_count.grid(row=0, column=4)
    

vcmd = (win.register(callback))  # register validate function


#############################################INPUT_FIELDS###############################################################
entry_new_unit = tk.Entry(frame, font=('Halvetica', 10))
entry_new_unit.bind('<KeyRelease>', check_entry)
entry_new_unit.grid(row=2, column=2)
if b is None:
    entry_new_unit.insert(0, "")
else:
    entry_new_unit.insert(0, b[0])
entry_new_id = tk.Entry(frame, width=10, font=('Halvetica', 10, "bold"), validate='all', validatecommand=(vcmd, '%P'))
entry_new_id.bind('<KeyRelease>', check_entry)
entry_new_id.grid(row=2, column=3)
if p is None:
    entry_new_id.insert(0, "")
else:
    entry_new_id.insert(0, str(int(p[1])))  # +1)))
entry_foto = tk.Entry(frame, validate='all', validatecommand=(vcmd, '%P'))
entry_foto.grid(row=2, column=4)
entry_foto.bind('<KeyRelease>', check_entry)

######################################################BUTTONS#########################################################
button_new_unit_from_cam = tk.Button(frame, text="3)Сделать снимки с камеры", command=create_new_image_unit,state="disabled")
button_new_unit_from_cam.grid(row=3, column=1)
button_db = tk.Button(frame, text='4)запомнить "лицо"', command=face_training, state="disabled")
button_db.grid(row=3, column=2)
button_open_db = tk.Button(win, text="Открыть БД", command=browse_image)
button_browse_model = tk.Button(win, text="1)выбор нейросети", command=browse_model)  # , state = "disabled")
button_browse_model.grid(row=0, column=0, stick="w")
button_start_cam1 = tk.Button(frame, text="включить камеру", command=start_camera1, state="disabled")
button_start_cam1.grid(row=3, column=3)
button_del_images = tk.Button(frame, text="удалить изображения из БД", command=del_files, state="disabled")
button_del_images.grid(row=3, column=4)
button_exit_programm = tk.Button(win, text="выход", command=exit_program)
button_exit_programm.grid(row=2, column=1)
button_stop_cameras = tk.Button(win, text="перезапуск программы", command=restart_program)
button_stop_cameras.grid(row=2, column=0, stick = "e")


#######################################################ENTRY_POINT######################################################
if __name__ == '__main__':
    start_message_help_info()
    win.mainloop()
