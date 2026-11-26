"""Code-to-label mappings. Labels follow the brief; reason labels follow the original notebook
(kept only for the documented-label column of the reason profile table)."""

GENDER = {1: "Male", 2: "Female"}

EDUCATION = {
    1: "No schooling (<7)", 2: "No schooling (7+)", 3: "Elementary", 4: "Middle school",
    5: "High school", 6: "Junior college", 7: "University", 8: "Master's", 9: "Doctorate",
}

REGION = {
    1: "Seoul", 2: "Gyeonggi", 3: "Gyeongnam", 4: "Gyeongbuk", 5: "Chungnam",
    6: "Gangwon & Chungbuk", 7: "Jeolla & Jeju",
}

# Inferred from the KoWePS coding convention and this data's profile (code 2:
# mostly older female heads living alone; code 5: young heads), not confirmed
# from a codebook. 0 and 9 are nonresponse (see NONRESPONSE).
MARRIAGE = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated",
            5: "Never married", 6: "Other"}

REASON = {
    1: "no capable", 2: "in military service", 3: "studying in school",
    4: "prepare for school", 5: "prepare to apply job", 6: "house worker",
    7: "caring kids at home", 8: "nursing", 9: "giving-up economic activities",
    10: "no intention to work", 11: "others",
}

# Major groups: first digit of the Korean Standard Classification of Occupations.
# 4-digit codes starting with "10" (1011 military officer, 1012 long-term
# sergeant) are armed forces, coded "A"; 9999 is an unknown code (missing).
OCC_MAJOR = {"1": "Managers", "2": "Professionals", "3": "Clerks", "4": "Service workers",
             "5": "Sales workers", "6": "Agriculture and fishery", "7": "Craft and trades",
             "8": "Machine operators", "9": "Elementary occupations", "A": "Armed forces"}

NONRESPONSE = {
    "marriage": {0, 9},
    "religion": {9},
    "company_size": {99},
    "reason_none_worker": {0, 99},
}
