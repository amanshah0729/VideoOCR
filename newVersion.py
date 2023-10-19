import cv2
import easyocr
min_conf = .3
video_path = 'FriendsLikeThat.mov'  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)
reader = easyocr.Reader(['en'])
overallWords = []
trackers = []

def newTrackers(resized_frame):
    print("none")
    results = reader.readtext(resized_frame)

    for (bbox, text, prob) in results:

        if prob >= min_conf:
            print("bbox")
            print(bbox)
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            x,y,w,h = x1, y1, x2 - x1, y2 - y1


            init_bbox = (int(x), int(y), int(w), int(h))
            print("heres the bounding box")
            print(init_bbox)
            #print(len(current_frame))
            overallWords.append(text)
            tracker = cv2.TrackerCSRT_create()

            # Initialize the tracker with the ROI
            tracker.init(resized_frame, init_bbox)

            # Add the tracker and ROI to the list
            trackers.append((tracker, bbox))
frame_counter = 0
while True:
    frame_counter += 1
    ret, frame = cap.read()
    if not ret:
        
        break
    if frame_counter % 5 == 0:
        
        resized_frame = cv2.resize(frame, (320, 320))
        if len(trackers) == 0:
            newTrackers(resized_frame)
        else:
            for tracker, bbox in trackers:
                success, new_bbox = tracker.update(resized_frame)

                if success:
                    #print("success, i tracked it")
                    x, y, w, h = [int(v) for v in new_bbox]
                    cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    #new code test below
                    
                    roi = resized_frame[y:y + h, x:x + w]

                    # Perform OCR on the cropped region
                    try:
                        results = reader.readtext(roi)
                        probs = []
                        for (Smallbbox, text, prob) in results:
                            # Print the text and its bounding box within the ROI
                            print(f"Text: {text}, Bounding Box: {Smallbbox}")
                            probs.append(prob)
                            
                        if all(prob < min_conf / 3 for prob in probs):
                            print("could not update tracker, lost it")
                            trackers.remove((tracker, bbox))
                            break
                    except:
                        pass

                    # Process the OCR results

                else:
                    print("could not update tracker, lost it")
                    trackers.remove((tracker, bbox))
                    #remove tracker from list, get the words it had

        cv2.imshow("Multi-object Tracking", resized_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# Release video capture object
cap.release()
cv2.destroyAllWindows()
print(overallWords)
