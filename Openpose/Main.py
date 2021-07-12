"""
File: Main.py

Authors: Jinwoo Jeong <jw.jeong@keti.re.kr>
         Sungjei Kim <sungjei.kim@keti.re.kr>
         Seungho Lee <seunghl@keti.re.kr>

The property of program is under Korea Electronics Technology Institute.
For more information, contact us at <jw.jeong@keti.re.kr>.
"""

def draw_fps(img, fps):
    tl = (
        round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = [0, 0, 0]
    #c1, c2 = (0, 0), (101, 101)
    #cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    tf = max(tl - 1, 1)  # font thickness
    label = "fps: " + str(float("{:.2f}".format(fps)));
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c1 = (0, t_size[1] + 15)
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 15
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled

    c1 = (0, t_size[1] + 3)
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] + 3
    cv2.putText(
        img,
        label,
        (c1[0], c1[1]),
        0,
        tl / 3,
        [225, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )

def preprocess(image):

def main():

if __name__ == "__main__":
    main()