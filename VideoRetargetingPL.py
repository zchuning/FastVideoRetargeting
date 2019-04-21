import numpy as np
import cv2

def stdCompensation(n):
    return 0.15 * np.sqrt(((255 - 255 / n) * (255 - 255 / n) + (n - 1) * ((255 / n) * (255 / n))) / n)

def printSeam(imgOrg, ce_indicator):
    img = np.copy(imgOrg)
    height, width = img.shape[0:2]
    for i in range(0, width):
        img[height - 1, i] = (0, 0, 255)
        itr = ce_indicator[height - 1, i]
        for j in range(1, height - 1):
            img[height - j - 1, itr] = (0, 0, 255)
            itr = ce_indicator[height - j - 1, itr]

    cv2.imshow("image", img)
    cv2.waitKey(0)

# The energy function takes two arguments, frame and previous frame
# The function weights ordinary Sobel map and difference to assign moving objects more energy in order to preserve them
# Exactly how they are weighted affects the threshold of std
def compute_energy(img, prev):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayp = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, grayp)
    # ret, diff = cv2.threshold(cv2.absdiff(gray, grayp), 0, 255, cv2.THRESH_BINARY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    sobel_weighted = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    # Motion weights more than Sobel map
    return cv2.addWeighted(sobel_weighted, 0.4, diff, 0.6, 0)

def removeSeamInArr(buffer, energy, nwidth):
    height, width = energy.shape[:2]
    assert (width >= 4), "Width cannot be less than 4"

    num_seam = width - nwidth
    assert (num_seam > 0), "Number of seam cannot be negative, check image size and input"

    rem_count = 0
    # Remove vertical seams
    # Need to remove width-nwidth seams
    while rem_count < num_seam:
        assert (energy.shape[1] == width), "Not the correct width"

        # Create a table of cumulative energy
        ce = np.zeros((height, width), dtype=np.int32)
        ce_indicator = np.zeros((height, width), dtype=np.int32)
        pt_lb = np.zeros((height, width), dtype=np.uint8)  # Parental label, enabling multiple removal

        ce[0, :] = energy[0, :]
        ce_indicator[0, :] = -1  # -1 means the head of table
        pt_lb[0, :] = range(0, width)

        for i in range(1, height):  # Starting from second row

            # Treating leftmost & rightmost pixel separately
            ce[i, 0] = energy[i, 0] + min(ce[i - 1, 0], ce[i - 1, 1])
            ce_indicator[i, 0] = np.argmin([ce[i - 1, 0:2]])
            pt_lb[i, 0] = pt_lb[i - 1, np.argmin([ce[i - 1, 0:2]])]

            ce[i, width - 1] = energy[i, width - 1] + min(ce[i - 1, width - 2], ce[i - 1, width - 1])
            ce_indicator[i, width - 1] = width - 2 + np.argmin([ce[i - 1, width - 2:width]])
            pt_lb[i, width - 1] = pt_lb[i - 1, width - 2 + np.argmin([ce[i - 1, width - 2:width]])]

            l = ce[i - 1, 0:width - 2]
            m = ce[i - 1, 1:width - 1]
            r = ce[i - 1, 2:width]

            # Assigning the cumulative energy
            ce[i, 1:width - 1] = energy[i, 1:width - 1] + np.minimum(np.minimum(l, m), r)

            # A bunch of logic, what they do is top secret
            llem = np.less_equal(l, m)
            mler = np.less_equal(m, r)
            ller = np.less_equal(l, r)
            left_least = np.logical_or(np.logical_and(llem, mler),
                                       np.logical_and(np.logical_and(llem, np.logical_not(mler)), ller))
            middle_least = np.logical_and(np.logical_not(llem), mler)

            for j in range(1, width - 1):
                if left_least[j - 1]:
                    ce_indicator[i, j] = j - 1
                    pt_lb[i, j] = pt_lb[i - 1, j - 1]
                elif middle_least[j - 1]:
                    ce_indicator[i, j] = j
                    pt_lb[i, j] = pt_lb[i - 1, j]
                else:
                    ce_indicator[i, j] = j + 1
                    pt_lb[i, j] = pt_lb[i - 1, j + 1]

        # Seam removal
        count = height - 1
        rem = []
        old = pt_lb[-1, 0]
        minidx = 0
        for i in range(1, width):
            if pt_lb[-1, i] != old:  # If it has another parent, push the last min to the array
                old = pt_lb[-1, i]
                rem.append(minidx)
                minidx = i
                continue
            if ce[-1, i] < ce[-1, minidx]:
                minidx = i

        rem.append(minidx)
        rem_num = len(rem)
        rem_count += rem_num
        if rem_count >= num_seam:
            rem_num -= rem_count - num_seam
            rem = rem[0:rem_num]

        # Trace back and remove every point in the seam
        while count > 0:
            for i in range(0, rem_num):
                buffer[:, count, rem[i] - i:width - 1 - i] = buffer[:, count, rem[i] - i + 1: width - i]
            rem[:] = ce_indicator[count, rem[:]]
            count = count - 1
        buffer = buffer[:, :, 0:-rem_num]

        # Update variables
        width = width - rem_num
        # Update energy, comment out for faster performance and less accuracy
        engs = np.stack([compute_energy(buffer[0], buffer[0])], axis=0)
        for idx in range(1, len(buffer)):
            eng = compute_energy(buffer[idx], buffer[idx-1])
            engs = np.append(engs, [eng], axis=0)
        energy = np.average(engs, axis=0)
    return buffer


def getBufferAndRemoveSeam(cap, nwidth):
        # Dynamic buffer + partial recalculation
        ret, frame = cap.read()
        if not ret:
            return None, None
        # The buffer here is a container of original frames
        buffer = np.stack([frame], axis=0)
        eng = compute_energy(frame, frame)
        # engs is a container of energy maps
        engs = np.stack([eng], axis=0)
        meanOfStd = 0.0
        counter = 2
        # The threshold is empirical. It is relevant to the weights in energy function.
        while meanOfStd < stdCompensation(counter):
            ret, frame = cap.read()
            if not ret:
                break
            eng = compute_energy(frame, buffer[-1])
            engs = np.append(engs, [eng], axis=0)
            buffer = np.append(buffer, [frame], axis=0)

            # First calculate the standard deviation of each pixel in an image
            # Then calculate the mean of the standard deviations
            meanOfStd, std = cv2.meanStdDev(engs.std(axis=0))
            counter += 1

        avgEng = np.average(engs, axis=0)
        return len(engs), removeSeamInArr(buffer, avgEng, nwidth)


def videoRetarget(name, outn, a):
    cap = cv2.VideoCapture(name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    nwidth = int(a * width)
    assert(nwidth < width), "New width cannot be greater than original width"

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (nwidth, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(outn, frc, fps, size, 1)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    i = 0
    while i < frame_count:
        # Acquire the buffer of processed frames
        ret, buffer = getBufferAndRemoveSeam(cap, nwidth)
        if ret is None:
            break
        else:
            i += ret
            print "Removing seam " + str(i) + " of " + str(frame_count)
            for fr in buffer:
                out.write(fr)

    cap.release()