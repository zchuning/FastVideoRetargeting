import numpy as np
import cv2

def printSeam(imgOrg, ce_indicator):
    img = np.copy(imgOrg)
    height, width = img.shape[0:2]
    for i in range(0, width):
        img[height - 1, i] = (0, 0, 255)
        itr = ce_indicator[height - 1, i]
        for j in range(1, height - 1):
            img[height - j - 1, itr] = (0, 0, 255)
            itr = ce_indicator[height - j - 1, itr]

    cv2.imwrite("./Seams.jpg", img)
    cv2.imshow("image", img)
    cv2.waitKey()


def compute_energy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)


def seamCarverVertical(img, nwidth):
    height, width = img.shape[:2]
    assert (width >= 4), "Width cannot be less than 4"

    num_seam = width - nwidth  # Number of iterations
    assert (num_seam > 0), "Number of seam cannot be negative, check image size and input"

    rem_count = 0
    # Remove vertical seams
    # Need to remove width-nwidth seams
    while rem_count < num_seam:
        assert (img.shape[1] == width), "Not the correct width"

        energy = compute_energy(img)

        # Create a table of cumulative energy
        ce = np.zeros((height, width), dtype=np.int32)
        ce_indicator = np.zeros((height, width), dtype=np.int32)
        pt_lb = np.zeros((height, width), dtype=np.uint8)  # Parental label, enabling multiple removal

        ce[0, :] = energy[0, :]
        ce_indicator[0, :] = -1  # -1 means the head of table because of line 30
        pt_lb[0, :] = range(0, width)

        for i in range(1, height):  # For each row starting from second row

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
            # Actually the right_least is redundant
            llem = np.less_equal(l, m)
            mler = np.less_equal(m, r)
            ller = np.less_equal(l, r)
            left_least = np.logical_or(np.logical_and(llem, mler),
                                       np.logical_and(np.logical_and(llem, np.logical_not(mler)), ller))
            middle_least = np.logical_and(np.logical_not(llem), mler)
            right_least = np.logical_or(np.logical_and(np.logical_not(llem), np.logical_not(mler)),
                                        np.logical_and(np.logical_and(llem, np.logical_not(mler)),
                                                       np.logical_not(ller)))

            for j in range(1, width - 1):
                if left_least[j - 1]:
                    ce_indicator[i, j] = j - 1
                    pt_lb[i, j] = pt_lb[i - 1, j - 1]
                elif middle_least[j - 1]:
                    ce_indicator[i, j] = j
                    pt_lb[i, j] = pt_lb[i - 1, j]
                elif right_least[j - 1]:
                    ce_indicator[i, j] = j + 1
                    pt_lb[i, j] = pt_lb[i - 1, j + 1]

        # printSeam(img, ce_indicator)
        # cv2.waitKey()

        # Seam removal
        count = height - 1
        rem = []
        old = pt_lb[-1, 0]
        minidx = 0
        for i in range(1, width):
            if pt_lb[-1, i] != old: # If it has another parent, push the last min to the array
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
                img[count, rem[i] - i:width - 1 - i] = img[count, rem[i] - i + 1: width - i]
            rem[:] = ce_indicator[count, rem[:]]
            count = count - 1
        img = img[:, 0:-rem_num]

        # Uncomment to display internal process
        # cv2.imshow("Result", img)
        # cv2.waitKey(1)

        # Update variables
        width = width - rem_num

    # Uncomment to display result
    # cv2.imshow("Result", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    cv2.imwrite("./result.jpg", img)
    return img