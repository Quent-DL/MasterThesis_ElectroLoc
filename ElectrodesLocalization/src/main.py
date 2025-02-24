import contacts
import segmentation


def main():
    # Fetching the contacts
    contacts = contacts.get_contacts()

    # Extract nb of electrodes
    # TODO remove hard-coded
    n_electrodes = 5

    # Segmenting contacts into electrodes
    electrodes = segmentation.segment_electrodes(contacts, n_electrodes)




if __name__ == '__main__':
    main()