#predict using the model
y_pred = resnet50.predict(test_generator).round()
y_test = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
y_pred = np.argmax(y_pred,axis=1)

#print classification report
print(classification_report(y_test,y_pred,target_names = class_labels))
