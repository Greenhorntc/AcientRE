from ReDataHandler import Re_data_handler
rehandler=Re_data_handler()
train,val,test=rehandler.get_raw_CHisIEC()
train=rehandler.change_CHisIEC_data(train)
val=rehandler.change_CHisIEC_data(val)
test=rehandler.change_CHisIEC_data(test)