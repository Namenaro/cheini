from easygui import multenterbox


def ask_user_for_name():
    msg = "Enter name for folder"
    title = "to save results"
    fieldNames = ["Name"]
    fieldValues = []  # we start with blanks for the values
    fieldValues = multenterbox(msg,title, fieldNames)

    # make sure that none of the fields was left blank
    while 1:
        if fieldValues is None: break
        errmsg = ""
        for i in range(len(fieldNames)):
            if fieldValues[i].strip() == "":
                errmsg += ('"%s" is a required field.\n\n' % fieldNames[i])
        if errmsg == "":
            break # no problems found
        fieldValues = multenterbox(errmsg, title, fieldNames, fieldValues)

    print("Reply was: %s" % str(fieldValues))
    return fieldValues[0]