import finrobot
print("finrobot:", dir(finrobot))
try:
    import finrobot.agents as fa
    print("finrobot.agents:", dir(fa))
except Exception as e:
    print("Error agents:", e)
    
try:
    import finrobot.functional as ff
    print("finrobot.functional:", dir(ff))
except Exception as e:
    print("Error functional:", e)
