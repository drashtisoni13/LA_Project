"""

MAT204: LINEAR ALGEBRA FINAL GROUP PROJECT

TITLE: APPLICATION OF LINEAR ALGEBRA IN AUDIO PROCESSING

GROUP NUMBER: 23

KALASH SHAH: AU1940287
URVISH MAKWANA: AU1940209
HINANSHI SUTHAR: AU1940266
DRASHTI SONI: AU1940058

"""
from tkinter import *
from tkinter import messagebox
top = Tk()

C = Canvas(top, bg="blue", height=3000, width=3000)

#add image file in main file and also image's path
filename = PhotoImage(file = "C:\\Users\\KALASH\\Desktop\\Final-Project\\img.png")
background_label = Label(top, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

C.pack()
top.mainloop

"""
Intro:
This is the first part of our programme, Audio Identification.
This programme can Identify entered input audio from the given database with the help of audio fingerprinting.
The database can store many audios but here we have just 2 songs because of RAM management issues in my device.
This programme's richer version can be said google search's "What's this song" and "Shazam".
Both of those work on the same Idea but, with larger databases.
THIS PART OF THE PROGRAMME IS NOT OUR ORIGINAL IDEA. THE ORIGINAL CREDITS ARE MENTIONED IN THE FINAL PROJECT REPORT.

Prerequisites:
The programme needs libraries/modules like 'scipy', 'time', 'os', 'numpy', 'pyaudio', 'wave', 'matplotlib', 'pydub', 'random'.
The recorded or noised sample of the input audio must be in .WAV form.
All the Audios needs to be in the database mentioned.
The sample audio wav also needs to be in separate folder/database.
All the audios in the database and the sample audio needs to be in .WAV format.

Workflow:
First, the programme reads the database and all the audios in the database.
Then, the program the sample audio from the different database is also read by the programme.
Programme works according to the Audio entered and uses the functions as needed such as clearing and hashing.
The audio is matched with the sample wav given.
The audio from the database which has more accurate results and more similarities is resulted as final match.
After the results declared, programme concludes there.
As much tries and different input audio wav and samples we have tried, the accuracy rate of the programme is 100%.

"""

#Importing all the needed libraries and modules:
import random
import pickle
import scipy.io.wavfile as wavfile
from scipy.signal import decimate, butter, filtfilt, spectrogram
from scipy.signal.windows import hamming
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
from skimage import util
from pydub import AudioSegment
import os
from shutil import copyfile
from appdirs import AppDirs
import pyaudio
import wave
import time


print("______________________________________________________________________________________________________")
print("|                                                                                                     |")
print("|                                                                                                     |")
print("|                                                                                                     |")
print("|                                            WELCOME!                                                 |")
print("|                                                                                                     |")
print("|                                                                                                     |")
print("|_____________________________________________________________________________________________________|")

print("\nPart 1: Audio Identification has started \n")

# Giving initial  values to variables:
Sample_Duration = 30
Default_Sampling_Rate = 44100
Sampling_Rate = 11025
Cutoff_Frequency = 5000
Samples_per_Window = 4096
Time_Resolution = Samples_per_Window / Sampling_Rate
Upper_Frequency_Limit = 600
# Frequency ranges for hashing:
Ranges = [40, 80, 120, 180, Upper_Frequency_Limit + 1]


#Defining a function to read the file and convert it to .WAV form using original framework:
def read_audio_file(filename):
    file_name, file_extension = os.path.splitext(filename)
    if file_extension != '.wav':
        filename = convert_to_wav(filename)
    else:
        copyfile(filename, os.path.join(os.path.join(
            AppDirs('Audio_ID').user_data_dir, 'Database'), os.path.basename(filename)))
    rate, data = wavfile.read(filename)
    return rate, data


#Function to convert from Stereo to Mono:
def stereo_to_mono(audiodata):
    return audiodata.sum(axis=1) / 2


#A function to convert format in WAV with the help of FFMPEG
def convert_to_wav(filename):
    try:
        source_parent = os.path.dirname(filename)
        filename = os.path.basename(filename)
        # song_title = filename.split('.')[0]
        # song_format = filename.split('.')[1]
        song_title, song_format = os.path.splitext(filename)
        exported_song_title = song_title + '.wav'
        original_song = AudioSegment.from_file(
            os.path.join(source_parent, filename), format=song_format[1:])
        original_song = original_song.set_channels(1)
        original_song = original_song.set_frame_rate(44100)
        exported_song = original_song.export(os.path.join(
            os.path.join(AppDirs('Audio_ID').user_data_dir, 'Database'), exported_song_title), format="wav")
        return os.path.join(
            os.path.join(AppDirs('Audio_ID').user_data_dir, 'Database'), exported_song_title)
    except IndexError:
        return None


#Butter Lowpass Function:
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


#Butter Lowpass Filter Function, returns filtered data:
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


#Getting decimate values, downsample Signal:
def downsample_signal(data, factor):
    return decimate(data, factor)


#To apply window:
def apply_window_function(data, window_size, window_function):
    # (window_size,) ==> dimensions are window_size x 1
    windows = util.view_as_windows(data, window_shape=(window_size,), step=100)
    windows = windows * window_function
    return windows


#Getting FFT Demo:
def fft_demo(data, window_size, window_function):
    fft_data = fft(data[:window_size] * window_function)
    freq = fftfreq(len(fft_data), 1 / Sampling_Rate)
    return np.abs(fft_data[:window_size // 2]), freq


#To get the FFT of the window, FFT on a single window:
def fft_one_window(window, window_size):
    fft_data = fft(window)
    freq = fftfreq(len(fft_data), 1 / Sampling_Rate)
    return np.abs(fft_data)[:window_size // 2], freq[:window_size // 2]


#Plotting unfiltered Spectrogram:
def plot_spectrogram(data, window_size, sampling_rate):
    freq, time, Spectrogram = spectrogram(data, fs=sampling_rate,
                                          window='hamming', nperseg=window_size,
                                          noverlap=window_size - 100, detrend=False,
                                          scaling='spectrum')
    f, ax = plt.subplots(figsize=(4.8, 2.4))
    ax.pcolormesh(time, freq / 1000, np.log10(Spectrogram), cmap="PuOr")
    ax.set_ylabel('Frequency [kHz]', fontsize=22)
    ax.set_xlabel('Time [sec]', fontsize=22)
    plt.title("Renai Circulation (Bakemonogatari OP)", fontsize=22)
    plt.show()


#To filter the spectrogram, returns filtered bins:
def filter_spectrogram(windows, window_size):
    # Init 2D list
    filtered_bins = [[0 for i in range(len(Ranges))] for j in range(
        len(windows))]

    for i in range(len(windows)):
        fft_data, freq = fft_one_window(windows[i], window_size)
        max_amp_freq_value = 0
        max_amp = 0
        current_freq_range_index = 0
        for j in range(len(fft_data)):
            if freq[j] > Upper_Frequency_Limit:
                continue

            # Reset max. amplitudes and bins for each band
            if current_freq_range_index != return_freq_range_index(freq[j]):
                current_freq_range_index = return_freq_range_index(freq[j])
                max_amp_freq_value = 0
                max_amp = 0
            if fft_data[j] > max_amp:
                max_amp = fft_data[j]
                max_amp_freq_value = freq[j]
            filtered_bins[i][current_freq_range_index] = max_amp_freq_value
    return filtered_bins


#To get Frequency Range Index, returns index value for given frequency:
def return_freq_range_index(freq_value):
    freq_range_index = 0
    while freq_value > Ranges[freq_range_index]:
        freq_range_index = freq_range_index + 1
    return freq_range_index


#Plotting filtered Spectrogram:
def plot_filtered_spectrogram(filtered_data):
    for window_index in range(len(filtered_data)):

        timestamp = np.array(
            [window_index] * len(filtered_data[window_index])) * Time_Resolution

        # Scatter plot of filtered bins
        # c => color of point, marker => shape of mark
        plt.scatter(timestamp, filtered_data[window_index], c='b', marker='.')

    # To force the graph to be plotted upto 512 even though our y values range
    # from 0 to 300
    plt.ylim(0, 512)

    # Below loop draws horizontal lines for each band
    for i in range(len(Ranges)):
        plt.axhline(y=Ranges[i], c='r')
    plt.show()


#Running the entire algorithm on a Audio, returns Filtered spectrogram data:
def song_recipe(filename):
    rate, audio_data = read_audio_file(filename)
    if audio_data.ndim != 1:  # Checks no. of channels. Some samples are already mono
        audio_data = stereo_to_mono(audio_data)
    filtered_data = butter_lowpass_filter(
        audio_data, Cutoff_Frequency, Default_Sampling_Rate)
    decimated_data = downsample_signal(
        filtered_data, Default_Sampling_Rate // Sampling_Rate)
    hamming_window = hamming(Samples_per_Window, sym=False)
    windows = apply_window_function(
        decimated_data, Samples_per_Window, hamming_window)
    filtered_spectrogram_data = filter_spectrogram(windows, Samples_per_Window)
    return filtered_spectrogram_data


#Here, we must assume that the recording is not the cleared one, so we had to include a fuzz factor to clear it.
#A filtered bin of a window generated by filter_spectrogram, returns hash value pf the particular bin:
def hash_window(filtered_bin):
    fuz_factor = 2  # If an error occurs
    return (filtered_bin[3] - (filtered_bin[3] % fuz_factor)) * 1e8 + (
            filtered_bin[2] - (filtered_bin[2] % fuz_factor)) * 1e5 + (
                   filtered_bin[1] - (filtered_bin[1] % fuz_factor)) * 1e2 + (
                   filtered_bin[0] - (filtered_bin[0] % fuz_factor))


#Following fuction modifies hash_dictionary to map data of the given song_id
def hash_song(song_id, filtered_bins, hash_dictionary):

    for i, filtered_bin in enumerate(filtered_bins):
        try:
            hash_dictionary[hash_window(filtered_bin)].append((song_id, i))
        except KeyError:
            hash_dictionary[hash_window(filtered_bin)] = [(song_id, i)]


#Creating a Hashmap of the filered bins:
def hash_sample(filtered_bins):
    sample_dictionary = {}
    for i, filtered_bin in enumerate(filtered_bins):
        try:
            sample_dictionary[hash_window(filtered_bin)].append(i)
        except KeyError:
            sample_dictionary[hash_window(filtered_bin)] = [i]
    return sample_dictionary


#To create a Database.
#Audio_to_ID: Maps song to generate IDs, ID_to_Audio: Maps IDs to Audio.
#Hash_Dictionary: Maps hash values to associated IDs and offset values.
def create_database(song_dir):
    if os.path.exists(os.path.join(AppDirs('Audio_ID').user_data_dir, 'Database')):
        pass
    else:
        os.mkdir(os.path.join(AppDirs('Audio_ID').user_data_dir, 'Database'))
    song_to_id = {}
    id_to_song = {}
    hash_dictionary = {}
    random_ids = random.sample(range(1000), len(os.listdir(song_dir)))
    for song_id, filename in zip(random_ids, os.listdir(song_dir)):
        print(filename)
        song_to_id[filename] = song_id
        id_to_song[song_id] = filename
        filtered_bins = song_recipe(os.path.join(song_dir, filename))
        hash_song(song_id, filtered_bins, hash_dictionary)
    with open(os.path.join(AppDirs('Audio_ID').user_data_dir, 'Database.pickle'), 'wb') as f:
        pickle.dump(song_to_id, f)
        pickle.dump(id_to_song, f)
        pickle.dump(hash_dictionary, f)
    print('\nDatabase created successfully!')
    return song_to_id, id_to_song, hash_dictionary


#Loading the Database in a serialized file, returns song_to_id, id_to_song, hash_dictionary:
def load_database():
    with open(os.path.join(AppDirs('Audio_ID').user_data_dir, 'Database.pickle'),
              'rb') as f:  # Load data from a binary file
        song_to_id = pickle.load(f)
        id_to_song = pickle.load(f)
        hash_dictionary = pickle.load(f)
    return song_to_id, id_to_song, hash_dictionary


#This function is the most important part of the project, which mathces both the samples.
#This is the Audio matching algorithm to find the correct Audio.
#Returns max_frequencies and max_frequencies_key which helps us to find the best match for the given input sample WAV.
def find_song(hash_dictionary, sample_dictionary, id_to_song):
    offset_dictionary = dict()
    for song_id in id_to_song.keys():
        offset_dictionary[song_id] = {}
    song_size = {}
    for song_id in id_to_song.keys():
        rate, data = wavfile.read(os.path.join(os.path.join(
            AppDirs('Audio_ID').user_data_dir, 'Database'), id_to_song[song_id]))
        song_size[song_id] = len(data) / rate
    for sample_hash_value, sample_offsets in sample_dictionary.items():
        for sample_offset in sample_offsets:
            try:
                for song_id, offset in hash_dictionary[sample_hash_value]:
                    try:
                        offset_dictionary[song_id][(
                                                           offset - sample_offset) // 1] += 1
                    except KeyError:
                        offset_dictionary[song_id][(
                                                           offset - sample_offset) // 1] = 1
            except KeyError:
                pass
    max_frequencies = {}
    for song_id, offset_dict in offset_dictionary.items():
        for relative_set, frequency in offset_dict.items():
            try:
                max_frequencies[song_id] = max(
                    max_frequencies[song_id], frequency)
            except KeyError:
                max_frequencies[song_id] = frequency
    max_frequencies_keys = sorted(
        max_frequencies, key=max_frequencies.get, reverse=True)
    return max_frequencies, max_frequencies_keys


#To convert into WAV format:
def batch_convert_to_wav():
    mp3_dir = "MP3 Database"
    for filename in os.listdir(mp3_dir):
        convert_to_wav(os.path.join(mp3_dir, filename), "Database")


#Recording Sample Recipe:
def record_sample_recipe(filename=os.path.join(AppDirs('Audio_ID').user_data_dir, 'sample.wav')):
    # Record in chunks of 1024 samples
    chunk = 1024
    # 16 bits per sample
    sample_format = pyaudio.paInt16
    #Number of channels 1
    channels = 1
    # Record at 44100 samples per second
    fs = 44100
    seconds = Sample_Duration
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)
    #Initializing an array to store the frames
    frames = []
    # Storing the data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()


#Playback recorded song
def playback_recorded_sample(filename=os.path.join(AppDirs('Audio_ID').user_data_dir, 'sample.wav')):
    # Set chunk size of 1024 samples per data frame
    chunk = 1024
    # Open the sound file
    wf = wave.open(filename, 'rb')
    # Create an interface to PortAudio
    p = pyaudio.PyAudio()
    # Open a .Stream object to write the WAV file to
    # 'output = True' indicates that the sound will be played rather than recorded
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    # Read data in chunks
    data = wf.readframes(chunk)
    # Play the sound by writing the audio data to the stream
    while data != '':
        stream.write(data)
        data = wf.readframes(chunk)
    # Close and terminate the stream
    stream.close()
    p.terminate()


#Identifying the Audio:
def identify_song(filename):
    if os.path.exists(os.path.join(AppDirs('Audio_ID').user_data_dir, 'Database.pickle')):
        pass
    else:
        print('Database has not been created. Please do so by running \'Audio_ID create-db\'')
    print("Loading database")
    print(".\n.\n.")
    song_to_id, id_to_song, hash_dict = load_database()
    print("Database loaded")
    print("\nProcessing...")
    filtered_bins_sample = None
    if filename is None:
        print("Recording started")
        record_sample_recipe()
        print("Recording finished")
        filtered_bins_sample = song_recipe(
            'sample.wav')  # Running our algorithm on the song
    else:
        filtered_bins_sample = song_recipe(
            filename)
    sample_dict = hash_sample(filtered_bins_sample)
    max_frequencies, max_frequencies_keys = find_song(
        hash_dict, sample_dict, id_to_song)
    count = 0
    print('\nIdentified closest match!!!')
    print('\nResults:')
    for song_id in max_frequencies_keys:
        print(id_to_song[song_id], max_frequencies[song_id])
        count += 1
        if count == 1:
            break

# Now, creating a database in local folder
create_database("Audio_Identification_Songs_Database")

# Then, Input the audio which needs to be identified
identify_song("Audio_Identification_sample/sample.wav")

print("\nPart 1: Audio Identification Completed Successfully!\n")


"""

Intro:
2nd part of our project
This programme can be used in application of generation amplitude of waves.
Here, the used gets both the things, the values of amplitude at each point and also the graph of amplitude(Which is normally used)
And also for getting the value of notes of the audio in the form of matrix.
This Programme generates a separate excel file of amplitudes in the folder/path provided in the code.
And the value of notes and matrix in generated in the programme environment itself.

Prerequisites:
To get the expected output, libraries needed are 'scipy', 'time', 'os', 'numpy'.
The recorded or noised sample of the input audio must be in .WAV form.
For amplitude part, the audio must be in the folder/path which is mentioned in the code.
Input Audio for the matrix part, must be in same database or same folder with this python file.

Workflow:
First, the start time and end time is recorded and difference is calculated to get the final duration of the programme:
Then, the program asks for the input audio folder where input audios are kept in .WAV form.
The number of audio is counted (Here for this demo, we have just included one sample audio).
Then the program reads the files and generates the values in 1D array.
Those values of amplitude are saved in a new CSV file in a folder.
Then, the final result comes with the number of audios converted and time taken.
After this, another time the audio file is opened and the array with the values of notes is printed in array format.
The programme concludes after printing the array.

"""

#Importing needed libraries:
from scipy.io.wavfile import read
import numpy
from os import listdir
import os
import time

print("\nPart 2: Wave Amplitude and Matrix Conversion has started!\n")

# Recording the start time with the help of time module to calculate the time:
Start_Time = time.time()

# Printing the number of audios in the provided path/folder:
print("Number of audios in the folder = %d" % len(listdir("wave_amplitude_sample_folder/")))

Audio_Completed = 0

# Browsing all the .wav file in the folder mentioned:
for Audio_File in os.listdir("wave_amplitude_sample_folder/"):

    # Ensuring that the processed file has .wav extension:
    if Audio_File.endswith("sample_amplitude.wav"):

        # Reading a file:
        wave_file_id = read("wave_amplitude_sample_folder/" + Audio_File)

        # Transforming the file wave in a 1D numerical array:
        amplitude = numpy.array(wave_file_id[1], dtype=float)

        # Saving the array in the 1st column of a CSV file:
        numpy.savetxt("CSV_amplitude/" + os.path.splitext(Audio_File)[0] + ".csv", amplitude, delimiter=",")

        # This won't repeat the loop here as we have just 1 audio r/n in the folder.
        Audio_Completed = Audio_Completed + 1

        # Display the process current state
        print("Converting " + Audio_File + " ---> " + os.path.splitext(Audio_File)[
            0] + ".CSV_amplitude: Completed %d / %d" % (Audio_Completed, len(listdir(
            "wave_amplitude_sample_folder/"))))

# Printing the number of audios converted and the time taken:
print("The %d file(s) processing lasted for %s seconds." % (len(listdir("wave_amplitude_sample_folder/")), (time.time() - Start_Time)))

print("\nThe CSV file hase been created and saved in the database!")

# To get the matrix of the notes for the inserted audio:
a= read("2. Audio.wav")
a=numpy.matrix(a[1])
for i in a:
    # Printing the array of notes:
    print("\nThe array for the notes of given audio is", end=" ")
    print(a)
    if a.any():
         break


print("\nPart 2: Wave Amplitude and Matrix Conversion is Successful!\n")


"""

Intro:
This is the part 3 of our project, Noise Reduction.
With the help of this programme, we can remove unnecessary noise from the sample and get the cleared audio.
This programme can not completely remove the noise, but can make it better for sure.

Prerequisites:
Needed libraries and modules are 'contextlib', 'math', 'wave', 'numpy'.
Input Audio needs to be in .WAV format
Input Audio must be in same database or same folder with this python file

Workflow:
This programme takes an input audio from the user in .WAV format which is in the folder/path mentioned in the code.
The data of the wave is interpreted and channels are checked.
Raw audio/noise is reduced from multi channels
The frequency is saved with moving average.
Final output wave with reduced noise is saved in the same directory.
"""

#Importing needed libraries:
import contextlib
import math
import wave
import numpy as np

print("\nPart 3: Noise Reduction has started!")

Input_name = '3. Noiced Sample.wav'
Output_name = '3. Noice Filtered.wav'

cutOffFrequency = 1000.0

# Moving average or running mean, source mentioned in the report:
def running_mean(x, windowSize):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize


# Interpreting the data of WAV file, source mentioned in the report:
def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved=True):
    if sample_width == 1:
        dtype = np.uint8  # unsigned char
    elif sample_width == 2:
        dtype = np.int16  # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")
    channels = np.frombuffer(raw_bytes, dtype=dtype)

    # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
    if interleaved:
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
    else:
        channels.shape = (n_channels, n_frames)
    return channels

with contextlib.closing(wave.open(Input_name, 'rb')) as cntb:
    sampleRate = cntb.getframerate()
    ampWidth = cntb.getsampwidth()
    nChannels = cntb.getnchannels()
    nFrames = cntb.getnframes()

    # Extract Raw Audio from multi-channel Wav File
    signal = cntb.readframes(nFrames * nChannels)
    cntb.close()
    channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)

    # Cutoff Frequency of moving average filter, getting window size:
    freqRatio = (cutOffFrequency / sampleRate)
    N = int(math.sqrt(0.196196 + freqRatio ** 2) / freqRatio)

    # Using moving average (only on first channel)
    filtered = running_mean(channels[0], N).astype(channels.dtype)
    wav_file = wave.open(Output_name, "w")
    wav_file.setparams((1, ampWidth, sampleRate, nFrames, cntb.getcomptype(), cntb.getcompname()))
    wav_file.writeframes(filtered.tobytes('C'))
    wav_file.close()

print("The modified and cleared audio wav has been created and saved in the database!")
print("\nPart 3: Noise Reduction is successful!\n")


"""

Intro:
This is the last, 4th part of our Project, Converting Audio from 2D to 3D.
This can be used to get the audio in 3rd Dimension.

Prerequisites:
Needs to Install ffmpeg
Needed libraries and modules, pydub and math.
Input Audio needs to be in .mp3 format
Input Audio must be in same database or same folder with this python file

Workflow:
This programme takes a input audio from the user in .mp3 format which is in the folder.
The input audio is cut in pieces
Then, those pieces are inverted and again merged
The merged new audio is saved in the same database, folder with the new name.

"""

#Importing needed libraries and modules:
from pydub import AudioSegment
from math import *

print("\nPart 4: Music 3D has started!\n")

# Defining a function to get Cosine values of radians of Indexes:
def Cos_Pan(index):
    return cos(radians(index))


#Interval of seconds between the seconds
Interval = 0.2 * 1000


# Inserting the audio with proper path provided and Inverting the given input audio:
Primary_Audio = AudioSegment.from_mp3('4. Input2D_Audio.mp3')
Audio_Inverted = Primary_Audio.invert_phase()
Primary_Audio.overlay(Audio_Inverted)


#Splitting the audio in parts and pieces & Inverting pieces:
print("Splitting of the Audio Started: ")
Splitted_Audio = Splitted_Audio_Inverted = []
Audio_Start_Point = 0
while Audio_Start_Point + Interval < len(Primary_Audio):
    Splitted_Audio.append(Primary_Audio[Audio_Start_Point:Audio_Start_Point + Interval])
    Audio_Start_Point += Interval
if Audio_Start_Point < len(Primary_Audio) :
    Splitted_Audio.append(Primary_Audio[Audio_Start_Point:])
print("Splitting of the Audio Ended: ")
print("For the given Input Audio total Pieces are: " + str(len(Splitted_Audio)))


#Now merging the Splitted pieces in a new song:
New_Audio = Splitted_Audio.pop(0)
Pan_Index = 0
for Piece in Splitted_Audio:
    Pan_Index += 5
    Piece = Piece.pan(Cos_Pan (Pan_Index))
    New_Audio = New_Audio.append(Piece, crossfade=Interval/50)


# Now, Saving the final output as the 3rd Dimensional Audio:
Output_File = open("4. Output3D_Audio.mp3", 'wb')
New_Audio.export(Output_File, format='mp3')
#The output audio is in 3rd Dimension, Enjoy your 2D song in 3D with this programme!!

print("\nEntered audio has been converted to 3rd Dimension and saved!")

print("\nPart 4: Music 3D is successful!\n")

print("All the parts of programme are successful and all the outputs are given")

print("______________________________________________________________________________________________________")
print("|                                                                                                     |")
print("|                                                                                                     |")
print("|                                                                                                     |")
print("|                                          Thank You!                                                 |")
print("|                                                                                                     |")
print("|                                                                                                     |")
print("|_____________________________________________________________________________________________________|")

