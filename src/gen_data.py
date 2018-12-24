from music21 import *
import numpy as np
import pickle
# Test completely parsing music from one piece
pieces_train = []
pieces_val = []
pieces_labels_train = []
pieces_labels_val = []
means, stds = [], []
H = 12
W = 20
note_map = {
		"A-": 1,
		"A" : 2,
		"A#": 3,
		"B-": 4,
		"B" : 5,
		"B#": 6,
		"C-": 7,
		"C" : 8,
		"C#": 9,
		"D-": 10,
		"D" : 11,
		"D#": 12,
		"E-": 13,
		"E" : 14,
		"E#": 15,
		"F-": 16,
		"F" : 17,
		"F#": 18,
		"G-": 19,
		"G" : 20,
		"G#": 21,
		"REST": 22,
}

reverse_note_map = {v: k for k, v in note_map.items()}


def get_valid_pieces(composer):
	total_pieces = corpus.getComposer(composer)
	vp = []
	"""
	for p in total_pieces:
	s = corpus.parse(p)
	if min([len(s.parts[i].recurse().getElementsByClass(["Note", "Rest"])) for i in range(min([len(s.parts), 4]))]) > (W * 3):
	vp.append(p)
	"""
	for p in total_pieces:
		valid_parts = []
		s = corpus.parse(p)
		parts = s.parts
		for par in parts:
			if len(par.recurse().getElementsByClass(["Note", "Rest"])) > (W*3):
				valid_parts.append(par)
		if len(valid_parts) >= 3:
			vp.append(p)
	return vp

def process_single_note(n):
	if n.isRest:
		return [22, 0, n.duration.quarterLength]
	else:
		return [note_map[n.name[0:2]], n.octave, n.duration.quarterLength]
#TODO map notes to (note, octave, duration)

def process_notes(notes):
	return list(map(lambda n: process_single_note(n), notes))

def calc_stats_channels(data):
	# N, H, W, C
	means = np.mean(data, axis=0)
	stds = np.std(data, axis=0)

	return (means, stds)

def extreme_durations(data):
	return np.min(data[:,:,:,2]), np.max(data[:, :, :, 2])

def extreme_octaves(data):
	return np.min(data[:,:,:, 1]), np.max(data[:, :, :, 1])


def preprocess_image(img, means, stds):
    """Preprocess an image (H, W, C)
    
    Subtracts the pixel mean and divides by the standard deviation.
    """
    img = img.astype(np.float32)
    img = (img - means)/ stds
    return img


def deprocess_image(img, means, stds, extremes):
	"""Undo preprocessing on an image and convert back to uint8."""
	img = np.squeeze(img)
	img = img * stds + means
	for i in range(img.shape[-1]):
		img[:, :, i] = np.clip(img[:, :, i], extremes[i][0], extremes[i][1])

	return img

def features_to_note(nId, octave, duration):
	if nId == 0:
		nId = 1
	noteWithPitch = reverse_note_map[nId]

	if noteWithPitch == "REST":
		r = note.Rest()

		r.duration.quarterLength = duration

		return r
	else:
		n = note.Note(noteWithPitch + str(octave), quarterLength=duration)
		n.storedInstrument = instrument.Piano()
		return n

	

def features_to_image(features, num_parts=4):
	# Features shape 1 x H x W x C

	features = np.squeeze(features)

	sc= stream.Score()

	for i in range(num_parts):
		p = stream.Part()
		p.offset = 0.0
		offset = 0.0
		notes = features[[j for j in range(i, H, num_parts)],:,:].reshape((-1, 3))
		#print(notes.shape)
		for j in range(notes.shape[0]):
			f = np.squeeze(notes[j])
			nId = int(round(f[0]))
			octave = int(round(f[1]))
			duration = round(f[2] * 4) / 4.0

			n = features_to_note(nId, octave, duration)
			p.append(n)
			offset += duration
			n.offset = offset
		sc.append(p)

	return sc


def process_corpus(pieces, labels, verbose=False):
	X = []
	y = []
	stride1 = 5
	stride2 = 10

	for index, piece in enumerate(pieces):
		s = corpus.parse(piece)
		valid_parts = []
		parts = s.parts

		for par in parts:
			if len(par.recurse().getElementsByClass(["Note", "Rest"])) > (W*3):
				valid_parts.append(par)

		valid_parts = valid_parts[:4]

		notes = []

		minLen = 9999999999999
		
		for i, p in enumerate(valid_parts):
			notes.append(p.recurse().getElementsByClass(["Note", "Rest"]))
			if len(notes[i]) < minLen:
				minLen = len(notes[i])

		for i in range(len(valid_parts)):
			notes[i] = notes[i][:minLen]

		count = 0
		num_parts = len(valid_parts)
		num_notes = minLen
		while count + (3*W) < num_notes:
			note_arr = np.zeros((H, W, 3))
			for i in range(num_parts):
				ns = np.split(np.array(process_notes(notes[i][count : count + W*3])), 3)
				ns = np.array(ns)
				if index == 0 and count == 0:
					print(ns.shape)
				note_arr[[i, i + 4, i + 8]] = ns
			if labels[index] == 0:
				count += stride1 #bach

			else:
				count += stride2
			X.append(note_arr)
			y.extend([labels[index]])
	return (X, y)



# print(len(X_total))

	


if __name__ == '__main__':
	
	num_train = 40
	num_val = 9
	valid_pieces = get_valid_pieces('monteverdi')

	ps = valid_pieces[0:num_train + num_val]
	print(len(ps) == num_train + num_val)
	pieces_train.extend(ps[0:num_train])
	pieces_val.extend(ps[num_train:])
	pieces_labels_train = pieces_labels_train + [2] * num_train
	pieces_labels_val.extend([2] * num_val)

	num_train = 450
	num_val = 35
	valid_pieces = pickle.load(open("valid_pal.txt", 'rb'))

	ps = valid_pieces[:num_train + num_val]
	pieces_train.extend(ps[:num_train])
	pieces_val.extend(ps[num_train:])
	pieces_labels_train = pieces_labels_train + [3] * num_train

	pieces_labels_val.extend([3] * num_val)

	



	valid_pieces = get_valid_pieces('bach')
	num_train = len(valid_pieces) - 5
	num_val = 5
	ps = valid_pieces[:num_train + num_val]
	pieces_train.extend(ps[:num_train])
	pieces_val.extend(ps[num_train:])
	pieces_labels_train = pieces_labels_train + [0] * num_train

	pieces_labels_val.extend([0] * num_val)

	num_train = 8
	num_val = 3
	valid_pieces = get_valid_pieces('beethoven')

	ps = valid_pieces[:num_train + num_val]
	pieces_train.extend(ps[:num_train])
	pieces_val.extend(ps[num_train:])
	pieces_labels_train = pieces_labels_train + [1] * num_train

	pieces_labels_val.extend([1] * num_val)

	num_train = 10
	num_val = 5
	valid_pieces = get_valid_pieces('mozart')

	ps = valid_pieces[:num_train + num_val]
	pieces_train.extend(ps[:num_train])
	pieces_val.extend(ps[num_train:])
	pieces_labels_train = pieces_labels_train + [4] * num_train

	pieces_labels_val.extend([4] * num_val)


	print(len(pieces_labels_val))
	print(len(pieces_val))
	print(len(pieces_labels_train))
	print(len(pieces_train))
	stride1 = 5
	stride2 = 10


	# Make boxes of 12 x 20 with a stride of 3
	X_train = []
	y_train = []
	X_val = []
	y_val = []

	X_train, y_train = process_corpus(pieces_train, pieces_labels_train)
	X_val, y_val = process_corpus(pieces_val, pieces_labels_val, verbose=True)
	X_total = np.array(X_train + X_val)
	X_train = np.array(X_train)
	y_train = np.array(y_train)
	X_val = np.array(X_val)
	y_val = np.array(y_val)
	means, stds = calc_stats_channels(X_train)
	minTime, maxTime = extreme_durations(X_total)
	minOct, maxOct = extreme_octaves(X_total)
	minNote, maxNote = (0, 22)
	extremes = [[minNote, maxNote], [minOct, maxOct], [minTime, maxTime]]

	

	print(len(X_total))
	corpus.parse(pieces_train[0]).write('musicxml', fp='testing_original.xml')
	s = features_to_image(X_train[0])
	s.write('musicxml', fp='testing.xml')

	for indx, img in enumerate(X_train):
		X_train[indx] = preprocess_image(img, means, stds)

	for indx, img in enumerate(X_val):
		X_val[indx] = preprocess_image(img, means, stds)

	
	pickle.dump(X_train, open( "data_train.txt", "wb" ))
	pickle.dump(y_train, open( "labels_train.txt", "wb"))

	pickle.dump(X_val, open("data_val.txt", "wb"))
	pickle.dump(y_val, open("labels_val.txt", "wb"))

	pickle.dump(means, open("means.txt", "wb"))
	pickle.dump(stds, open("stds.txt", "wb"))

	pickle.dump(extremes, open("extremes.txt", "wb"))
	
	

