import mido
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.misc

from matplotlib.colors import colorConverter
from matplotlib.backends.backend_agg import FigureCanvasAgg

class MidiFile(mido.MidiFile):
    def __init__(self, filename, sr = 10, verbose = True):
        mido.MidiFile.__init__(self, filename)
        self.verbose = verbose
        self.sr = sr
        self.meta = {}
        self._events = self._get_events()
        self._total_ticks = self._get_total_ticks()
        self._roll = self._get_roll()

    def get_roll_image(self):
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        ax.axis('off')
        ax.set_facecolor('black')

        # change unit of time axis from tick to second
        second = mido.tick2second(self.total_ticks, self.ticks_per_beat, self.get_tempo())
        if second > 10:
            x_label_period_sec = second//10
        else:
            x_label_period_sec = second/10
        x_label_interval = mido.second2tick(x_label_period_sec, self.ticks_per_beat, self.get_tempo())/self.sr

        #plt.xticks([int(x*x_label_interval) for x in range(20)], [round(x*x_label_period_sec, 2) for x in range(20)])
        #plt.yticks([y*16 for y in range(8)], [y*16 for y in range(8)])

        # build colors
        channel_nb = 16
        transparent = colorConverter.to_rgba('black')
        colors = [mpl.colors.to_rgba(mpl.colors.hsv_to_rgb((i/channel_nb, 1, 1)), alpha = 1) for i in range(channel_nb)]
        cmaps = [mpl.colors.LinearSegmentedColormap.from_list('my_cmap', [transparent, colors[i]], 128) for i in range(channel_nb)]

        # build color maps
        for i in range(channel_nb):
            cmaps[i]._init()
            # create your alpha array and fill the colormap with them.
            alphas = np.linspace(0, 1, cmaps[i].N + 3)
            # create the _lut array, with rgba values
            cmaps[i]._lut[:, -1] = alphas

        for i in range(channel_nb):
            try:
                ax.imshow(self.roll[i], origin = 'lower', interpolation = 'nearest', cmap = cmaps[i], aspect = 'auto')
            except IndexError:
                pass

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img = np.fromstring(canvas.tostring_rgb(), dtype = 'uint8')

        width, height = fig.get_size_inches()*fig.get_dpi()
        width = int(width)
        height = int(height)
        img = img.reshape(height, width, 3)

        plt.close(fig)
        return img

    def draw_roll(self, draw_colorbar = True):
        fig, ax = plt.subplots(1)
        ax.axis('equal')
        ax.set_facecolor('black')

        # change unit of time axis from tick to second
        second = mido.tick2second(self.total_ticks, self.ticks_per_beat, self.get_tempo())
        if second > 10:
            x_label_period_sec = second//10
        else:
            x_label_period_sec = second/10
        x_label_interval = mido.second2tick(x_label_period_sec, self.ticks_per_beat, self.get_tempo())/self.sr

        if self.verbose:
            print('Roll length in seconds: %i.' % second)
            print('X label tic length in seconds: %i.' % x_label_period_sec)
            print('X label tic length in ticks: %i.' % x_label_interval)

        # change scale and label of x and y axis
        plt.xticks([int(x*x_label_interval) for x in range(20)], [round(x*x_label_period_sec, 2) for x in range(20)])
        plt.yticks([y*16 for y in range(8)], [y*16 for y in range(8)])

        # build colors
        channel_nb = 16
        transparent = colorConverter.to_rgba('black')
        colors = [mpl.colors.to_rgba(mpl.colors.hsv_to_rgb((i/channel_nb, 1, 1)), alpha = 1) for i in range(channel_nb)]
        cmaps = [mpl.colors.LinearSegmentedColormap.from_list('my_cmap', [transparent, colors[i]], 128) for i in range(channel_nb)]

        # build color maps
        for i in range(channel_nb):
            cmaps[i]._init()
            # create your alpha array and fill the colormap with them.
            alphas = np.linspace(0, 1, cmaps[i].N + 3)
            # create the _lut array, with rgba values
            cmaps[i]._lut[:, -1] = alphas

        # draw piano roll and stack image on a1
        for i in range(channel_nb):
            try:
                ax.imshow(self.roll[i], origin = 'lower', interpolation = 'nearest', cmap = cmaps[i], aspect = 'auto')
            except IndexError:
                pass

        # draw color bar
        if draw_colorbar:
            colors = [mpl.colors.hsv_to_rgb((i/channel_nb, 1, 1)) for i in range(channel_nb)]
            cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap', colors, 16)
            a2 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            cbar = mpl.colorbar.ColorbarBase(
                a2,
                cmap = cmap,
                orientation = 'horizontal',
                ticks = list(range(16))
            )

        # show piano roll
        plt.draw()
        plt.show()

    def save_roll(self, dest_path):
        img = self.get_roll_image()
        scipy.misc.imsave(dest_path, img)
        return

        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        ax.axis('off')
        fig.imshow(img)
        fig.savefig(dest_path, facecolor = 'black')
        plt.close(fig)
        return
        # change unit of time axis from tick to second
        second = mido.tick2second(self.total_ticks, self.ticks_per_beat, self.get_tempo())
        if second > 10:
            x_label_period_sec = second//10
        else:
            x_label_period_sec = second/10
        x_label_interval = mido.second2tick(x_label_period_sec, self.ticks_per_beat, self.get_tempo())/self.sr
        plt.xticks([int(x*x_label_interval) for x in range(20)], [round(x*x_label_period_sec, 2) for x in range(20)])

        # change scale and label of y axis
        plt.yticks([y*16 for y in range(8)], [y*16 for y in range(8)])

        # build colors
        channel_nb = 16
        transparent = colorConverter.to_rgba('black')
        colors = [mpl.colors.to_rgba(mpl.colors.hsv_to_rgb((i/channel_nb, 1, 1)), alpha = 1) for i in range(channel_nb)]
        cmaps = [mpl.colors.LinearSegmentedColormap.from_list('my_cmap', [transparent, colors[i]], 128) for i in range(channel_nb)]

        # build color maps
        for i in range(channel_nb):
            cmaps[i]._init()
            # create your alpha array and fill the colormap with them.
            alphas = np.linspace(0, 1, cmaps[i].N + 3)
            # create the _lut array, with rgba values
            cmaps[i]._lut[:, -1] = alphas

        for i in range(channel_nb):
            try:
                ax.imshow(self.roll[i], origin = 'lower', interpolation = 'nearest', cmap = cmaps[i], aspect = 'auto')
            except IndexError:
                pass

        fig.savefig(dest_path, facecolor = 'black')
        plt.close(fig)

    def get_tempo(self):
        try:
            return self.meta['set_tempo']['tempo']
        except:
            # TODO why 500k?
            return 500000

    def get_chunk_generator(self, chunk_length):
        img = self.get_roll_image()
        for i in range(0, np.shape(img)[1], chunk_length):
            yield img[:, i:i + chunk_length]

    def _get_roll(self):
        length = self.total_ticks
        roll = np.zeros((16, 128, length//self.sr), dtype = 'int8')
        # use a register array to save the state (on/off) for each key
        note_register = [int(-1) for x in range(128)]
        # use a register array to save the state (program_change) for each channel
        timbre_register = [1 for x in range(16)]

        for idx, channel in enumerate(self.events):
            time_counter = 0
            volume = 100
            # Volume would change by control change event (cc) cc7 & cc11
            # Volume 0-100 is mapped to 0-127

            if self.verbose:
                print('channel', idx, 'start')

            for msg in channel:
                if msg.type == 'control_change':
                    if msg.control == 7:
                        volume = msg.value
                    if msg.control == 11:
                        # Change volume by percentage
                        volume = volume*msg.value//127

                if msg.type == 'program_change':
                    if self.verbose:
                        print('channel', idx, 'pc', msg.program, 'time', time_counter, 'duration', msg.time)
                    timbre_register[idx] = msg.program

                if msg.type == 'note_on':
                    if self.verbose:
                        print('on ', msg.note, 'time', time_counter, 'duration', msg.time, 'velocity', msg.velocity)
                    note_on_start_time = time_counter//self.sr
                    note_on_end_time = (time_counter + msg.time)//self.sr
                    intensity = volume*msg.velocity//127

					# When a note_on event *ends* the note starts to be played
					# Record end time of note_on event if there is no value in register
					# When note_off event happens, we fill in the color
                    if note_register[msg.note] == -1:
                        note_register[msg.note] = (note_on_end_time, intensity)
                    else:
					# When note_on event happens again, we also fill in the color
                        old_end_time = note_register[msg.note][0]
                        old_intensity = note_register[msg.note][1]
                        roll[idx, msg.note, old_end_time: note_on_end_time] = old_intensity
                        note_register[msg.note] = (note_on_end_time, intensity)

                if msg.type == 'note_off':
                    if self.verbose:
                        print('off', msg.note, 'time', time_counter, 'duration', msg.time, 'velocity', msg.velocity)
                    note_off_start_time = time_counter//self.sr
                    note_off_end_time = (time_counter + msg.time)//self.sr
                    note_on_end_time = note_register[msg.note][0]
                    intensity = note_register[msg.note][1]
					# fill in color
                    roll[idx, msg.note, note_on_end_time:note_off_end_time] = intensity
                    note_register[msg.note] = -1  # reinitialize register

                time_counter += msg.time

                # TODO: velocity -> done, but not verified
                # TODO: Pitch wheel
                # TODO: Channel - > Program Changed / Timbre catagory
                # TODO: real time scale of roll

            # If a note is not closed at the end of a channel, close it
            for key, data in enumerate(note_register):
                if data != -1:
                    note_on_end_time = data[0]
                    intensity = data[1]
                    note_off_start_time = time_counter//self.sr
                    roll[idx, key, note_on_end_time:] = intensity
                note_register[idx] = -1

        return roll

    def _get_total_ticks(self):
        max_ticks = 0
        for channel in range(16):
            ticks = sum(msg.time for msg in self.events[channel])
            if ticks > max_ticks:
                max_ticks = ticks
        return max_ticks

    def _get_events(self):
        if self.verbose:
            print(self)

        # There are more than 16 channel in midi.tracks. However there are only 16 channel related to "music" events.
        # We store music events of 16 channel in the list "events" with form [[ch1],[ch2]....[ch16]]
        # Lyrics and meta data used a extra channel which is not include in "events"

        events = [[] for x in range(16)]

        # Iterate all event in the midi and extract to 16 channel form
        for track in self.tracks:
            for msg in track:
                try:
                    channel = msg.channel
                    events[channel].append(msg)
                except AttributeError:
                    try:
                        if type(msg) != type(mido.UnknownMetaMessage):
                            self.meta[msg.type] = msg.dict()
                        else:
                            pass
                    except:
                        print('error', type(msg))

        return events

    @property
    def roll(self):
        return self._roll

    @property
    def total_ticks(self):
        return self._total_ticks

    @property
    def events(self):
        return self._events


if __name__ == '__main__':
    mid = MidiFile('test_file/imagine_dragons-believer.mid', verbose = False)
    mid = MidiFile('test_file/1.mid', verbose = False)

    # get the list of all events
    events = mid.events

    # get the np array of piano roll image
    roll = mid.get_roll()

    # draw piano roll by pyplot
    mid.draw_roll(draw_colorbar = False)
