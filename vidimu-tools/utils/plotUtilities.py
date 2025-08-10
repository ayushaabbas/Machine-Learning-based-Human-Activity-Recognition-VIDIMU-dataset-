import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import utils.signalProcessing as sp
import utils.fileProcessing as fp
from utils.opensimHelper import opensimTableToDataFrame

# ... previous functions ...

#IMUS MOT

def plotMotJointAnglesPerActivity(inpath,outpath,subjects,activity,activity_legend,motsignals,motrange,outputfilename=None):
    ncols = len(motsignals)
    nrows = len(subjects)

    fig,axes=plt.subplots(nrows,ncols,figsize=(ncols*6,nrows*3))
    prefixname = 'ik_'

    for i,subject in enumerate(subjects):
        dfmot = None
        found_file = False
        for trial in ["01","02","03","04","05"]:
            trialname = 'T'+trial
            motsubjacttrial = subject+"_"+activity+"_"+trialname
            motfilename = prefixname + motsubjacttrial + ".mot"
            inpathmotfull = os.path.join(inpath, subject, motfilename)

            if not os.path.exists(inpathmotfull):
                continue

            try:
                dfmot = opensimTableToDataFrame(inpathmotfull)
                found_file = True
                break  # use first existing trial
            except:
                continue

        if not found_file or dfmot is None:
            continue

        for j in range(ncols):
            if j > len(motsignals)-1:
                continue
            imujoint = motsignals[j]
            yaxis0, yaxis1 = motrange.get(imujoint, (-180, 180))
            if imujoint == '':
                continue
            try:
                jointangle_imus = dfmot[imujoint].to_numpy().flatten()[2:]
            except:
                continue

            X = np.arange(0, jointangle_imus.shape[0])
            color = 'k'
            if 'pelvis' in imujoint: color = 'r'
            if 'hip' in imujoint: color = 'g'
            if 'knee' in imujoint: color = 'b'
            if 'lumbar' in imujoint: color = 'r'
            if 'arm' in imujoint: color = 'g'
            if 'elbow' in imujoint: color = 'b'
            if 'wrist' in imujoint: color = 'r'

            axes[i][j].plot(X, jointangle_imus, color)
            axes[i][j].set_ylim([yaxis0, yaxis1])
            axes[i][j].set_title(motfilename + ' (' + str(imujoint) + ')')
            axes[i][j].set_ylabel("Degrees")
            axes[i][j].set_xlabel("Samples (50 Hz)")

    title = "Activity " + activity + ": " + activity_legend + " (one subject per row)"
    plt.suptitle(title, fontsize=18, verticalalignment='top', y=1.0)
    plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0)

    if outputfilename:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        plt.savefig(os.path.join(outpath, outputfilename+'.svg'), format='svg')
        plt.savefig(os.path.join(outpath, outputfilename+'.pdf'), format='pdf')
    plt.show()

def plotVideoPoseAllSubjectsOneActivity(inpath, outpath, subjects, activity, activity_legend, outputfilename=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import utils.fileProcessing as fp
    import utils.signalProcessing as sp
    import os

    ncols = 6
    nrows = len(subjects)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 3))
    for i, subject in enumerate(subjects):
        dfcsv = None
        for trial in ["01", "02", "03", "04", "05"]:
            trialname = "T" + trial
            csvsubjacttrial = f"{subject}_{activity}_{trialname}.csv"
            inpathfull = os.path.join(inpath, subject, csvsubjacttrial)
            if not os.path.exists(inpathfull):
                continue
            else:
                # dfcsv = fp.readCSV(inpathfolder, subject, activity, trialname)
                csvsubjacttrial = f"{subject}_{activity}_{trialname}.csv"
                infile = os.path.join(inpath, subject, csvsubjacttrial)

                if not os.path.exists(infile):
                    print(f"⚠️ No file: {infile}")
                    continue

                dfcsv = pd.read_csv(infile)

                break

        if dfcsv is None:
            print(f"⚠️ No data found for {subject} - {activity}")
            continue

        # Filtered joint angles
        arm_flex_r, arm_flex_l, elbow_flex_r, elbow_flex_l, knee_angle_r, knee_angle_l = fp.getMainJointAnglesFromCSV2(dfcsv)
        signals = [arm_flex_r, elbow_flex_r, knee_angle_r, arm_flex_l, elbow_flex_l, knee_angle_l]
        joint_names = ['R shoulder', 'R elbow', 'R knee', 'L shoulder', 'L elbow', 'L knee']
        colors = ['r', 'g', 'b', 'r', 'g', 'b']

        for j in range(ncols):
            signal_filtered = sp.applyMedianFilter(signals[j], 11)
            X = np.arange(len(signal_filtered))
            axes[i][j].plot(X, signal_filtered, color=colors[j])
            axes[i][j].set_title(f"{subject}_{activity}_{trialname} ({joint_names[j]})")
            axes[i][j].set_ylabel("Degrees")
            axes[i][j].set_xlabel("Samples (30 Hz)")

    title = f"Activity {activity}: {activity_legend} (one subject per row)"
    plt.suptitle(title, fontsize=18, verticalalignment='top', y=1.0)
    plt.tight_layout(pad=1.0)

    if outputfilename:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        plt.savefig(os.path.join(outpath, outputfilename + ".svg"), format='svg')
        plt.savefig(os.path.join(outpath, outputfilename + ".pdf"), format='pdf')

    plt.show()


import matplotlib.pyplot as plt

def plotRawQuaternionsPerActivity(inpath, outpath, subjects, activity, legend, imu_list, outputfilename):
    import matplotlib.pyplot as plt
    import os
    import pandas as pd

    ncols = len(imu_list)
    nrows = len(subjects)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 3))

    for i, subject in enumerate(subjects):
        file_found = False
        for trial in ["01", "02", "03", "04", "05"]:
            trialname = "T" + trial
            filename = f"{subject}_{activity}_{trialname}.raw"
            filepath = os.path.join(inpath, subject, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, skiprows=1, names=["sensor", "w", "x", "y", "z", "timestamp"])
                file_found = True
                break

        if not file_found:
            print(f"⚠️ No file found for {subject} - {activity}")
            continue

        for j, sensor in enumerate(imu_list):
            df_sensor = df[df['sensor'] == sensor]
            if df_sensor.empty:
                continue
            X = range(len(df_sensor))
            for quat in ['w', 'x', 'y', 'z']:
                axes[i][j].plot(X, df_sensor[quat], label=quat)
            axes[i][j].set_title(f"{subject}_{activity} ({sensor})")
            axes[i][j].legend()

    title = f"Activity {activity}: {legend} (one subject per row)"
    plt.suptitle(title, fontsize=18, verticalalignment='top', y=1.0)
    plt.tight_layout(pad=1.0)

    if outputfilename:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        plt.savefig(os.path.join(outpath, outputfilename + ".svg"), format='svg')
        plt.savefig(os.path.join(outpath, outputfilename + ".pdf"), format='pdf')

    plt.show()

