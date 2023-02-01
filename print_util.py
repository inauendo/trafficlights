import os

def underscore(string):
    return "\033[4m"+string+"\033[0m"

def green_section(string, start, end=0):
    tmp = list(string)
    if end == 0:
        tmp.append("\033[0m")
    else:
        tmp.insert(end, "\033[0m")
    tmp.insert(start, "\033[92m")
    return ''.join(tmp)

def replace_at(string, index, newchar):
    '''replaces the character in string at position index with the newchar'''
    tmp = list(string)
    length = len(newchar)
    if index+length != 0:
        tmp[index:(index+length)] = newchar
    else:
        tmp[index:] = newchar
    return ''.join(tmp)


def state_string(lanes, can_drive):

    #create strings for the lanes
    state = ["X" if entry > 0 else " " for entry in lanes]
    line1 = "_|{0}|{1}|{2}|       _".format(state[8], state[7], state[6])
    line2 = 15*" " + underscore("{0}".format(state[5]))
    line3 = 15*" " + underscore("{0}".format(state[4]))
    line4 = "_" + 14*" " + underscore("{0}".format(state[3]))
    line5 = underscore("{0}".format(state[9])) + 15*" "
    line6 = underscore("{0}".format(state[10])) + 15*" "
    line7 = underscore("{0}".format(state[11])) + 14*" " + "_"
    line8 = "        |{0}|{1}|{2}|".format(state[0], state[1], state[2])

    #edit strings according to can_drive
    if can_drive[0] == 1:
        line7 = replace_at(line7, -7, "|")
        line6 = replace_at(line6, -7, "|")
        line5 = replace_at(line5, -7, "|")
        line4 = replace_at(line4, 9, "|")
        line3 = replace_at(line3, 0, "----------")

    if can_drive[1] == 1:
        line1 = replace_at(line1, -5, "|")
        line2 = replace_at(line2, 11, "|")
        line3 = replace_at(line3, 11, "|")
        line4 = replace_at(line4, 11, "|")
        line5 = replace_at(line5, -5, "|")
        line6 = replace_at(line6, -5, "|")
        line7 = replace_at(line7, -5, "|")

    if can_drive[2] == 1:
        line7 = replace_at(line7, -3, "|")
        line6 = replace_at(line6, -3, "---")

    if can_drive[3] == 1:
        line4 = replace_at(line4, 4, "-----------")
        line5 = replace_at(line5, -12, "|")
        line6 = replace_at(line6, -12, "|")
        line7 = replace_at(line7, -12, "|")
        line8 = replace_at(line8, 4, "|")

    if can_drive[4] == 1:
        line3 = replace_at(line3, 0, "---------------")

    if can_drive[5] == 1:
        line1 = replace_at(line1, -5, "|")
        line2 = replace_at(line2, 11, "----")

    if can_drive[6] == 1:
        line2 = replace_at(line2, 6, "|")
        line3 = replace_at(line3, 6, "|")
        line4 = replace_at(line4, 6, "|")
        line5 = replace_at(line5, -10, "|")
        line6 = replace_at(line6, -10, "----------")

    if can_drive[7] == 1:
        line2 = replace_at(line2, 4, "|")
        line3 = replace_at(line3, 4, "|")
        line4 = replace_at(line4, 4, "|")
        line5 = replace_at(line5, -12, "|")
        line6 = replace_at(line6, -12, "|")
        line7 = replace_at(line7, -12, "|")
        line8 = replace_at(line8, 4, "|")

    if can_drive[8] == 1:
        line2 = replace_at(line2, 2, "|")
        line3 = replace_at(line3, 0, "---")

    if can_drive[9] == 1:
        line1 = replace_at(line1, -5, "|")
        line2 = replace_at(line2, 11, "|")
        line3 = replace_at(line3, 11, "|")
        line4 = replace_at(line4, 11, "|")
        line5 = replace_at(line5, -15, "-----------")

    if can_drive[10] == 1:
        line6 = replace_at(line6, -15, "---------------")

    if can_drive[11] == 1:
        line7 = replace_at(line7, -15, "----")
        line8 = replace_at(line8, 4, "|")

    #color in drivepaths
    line1 = green_section(line1, 8, -1)
    line2 = green_section(line2, 0, 15)
    line3 = green_section(line3, 0, 15)
    line4 = green_section(line4, 1, 15)
    line5 = green_section(line5, -16)
    line6 = green_section(line6, -16)
    line7 = green_section(line7, -16, -1)
    line8 = green_section(line8, 0, 8)

    return line1+"\n"+line2+"\n"+line3+"\n"+line4+"\n"+line5+"\n"+line6+"\n"+line7+"\n"+line8
