import sys
# move to velocity estimation folder
sys.path.insert(1, '/Users/gillianminnehan/Documents/macbookpro_docs/umich/eecs507/final-proj/indoor-localization/dev/velocity_estimation')
import user_velocity as uv

def main():
    # TODO: test RSSI accuracy on idle points
    # print("Testing RSSI accuracy on idle points...")

    # TODO: test RSSI accuracy on paths
    print("Testing RSSI accuracy on paths...")
    
    # iterate over 3 test paths
        # read in results csv (resultsx.csv) (get timestamp, est coord)
        # read in ground truth csv (test_velx.csv) (get timestamp, est coord)
        # match up timestamps, print each error to an array, compute avg error

    # TODO: test velocity estimation
    # print("Testing velocity estimation...")

    return 0

if __name__ == "__main__": 
    main()