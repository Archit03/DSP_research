encrypted_text = "E TZ 4 2 4 2 E T T E S H E E T T T T T T T T T T NO 8 4 2 4 2 I E A T T E S I H T T T T T T T T ET 2 8 4 2 8 4 2 E I E I T T T S E TMI 4 2 8 4 2 8 4 2 E 5 T T T I E H T T TI 4 2 8 4 2 4 2 E I T T T E I E S T T TI 4 2 8 4 2 4 2 E S T T T I I MD 4 2 E 8 4 2 8 4 2 H I T T T E I I E T T TI 4 2 4 2 8 4 Ŭ H E S E E <AS> E T T E E E E E I E I E 5 E T E C Ç Ŭ W B E T I T Ŝ B NT 5 E I T 5 E I I L T E E E I I H I I S H E U <BT> S H E E E E N E Ŭ K I S I E E E S E T E E I I 4 U Ŭ E S I N 1 E Ð K <AR> I I E N Ç A A T U E S S E A A V I A T U D U Y Z E I A X L Ç Š G U E K T W <BT>I E E E Ç H E E Ø"
frequency_table = {}
for letter in encrypted_text:
    if letter.isalpha():
        letter = letter.upper()
        if letter in frequency_table:
            frequency_table[letter] += 1
        else:
            frequency_table[letter] = 1

sorted_frequency_table = sorted(frequency_table.items(), key=lambda x: x[1], reverse=True)
print("Frequency table:", sorted_frequency_table)

most_common_encrypted_letters = [item[0] for item in sorted_frequency_table]
most_common_original_letters = ["E", "T", "A", "O", "I", "N", "S", "H", "R", "D", "L", "U"]

substitution_dict = {}
for i in range(len(most_common_encrypted_letters)):
    substitution_dict[most_common_encrypted_letters[i]] = most_common_original_letters[i]

decrypted_text = ""
for letter in encrypted_text:
    if letter.isalpha():
        letter = letter.upper()
        decrypted_text += substitution_dict[letter]
    else:
        decrypted_text += letter

print("Decrypted text:", decrypted_text)
