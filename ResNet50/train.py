checkpoint_filepath = '/tmp/checkpoint'
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor="val_loss",
                            mode="min",
                            save_best_only = True,
                            verbose=1)

earlystopping = EarlyStopping(monitor='val_loss',min_delta = 0, patience = 5, verbose = 1, restore_best_weights=True)

# loss = tf.keras.losses.sparse_categorical_crossentropy()
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile the model
resnet50.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


#Train model
history = resnet50.fit(train_generator, epochs=50, validation_data=val_generator, callbacks=[checkpoint,earlystopping])
