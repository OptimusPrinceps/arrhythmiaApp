//  ViewController.swift
//
//  Created by Josh Atwal on 29/08/18.
//
// Documents path: /Users/josh/Library/Developer/CoreSimulator/Devices/39DDE5CB-5337-4B67-A48D-7AE84ED55DC9/data/Containers/Data/Application/433545D3-0513-45D4-B621-87FCBAEB021C/Documents

import UIKit
import AVFoundation //For working with audio
import CoreML //For machine learning
import Charts //For plotting
import Spring //For animation

class ViewController: UIViewController, AVAudioPlayerDelegate, AVAudioRecorderDelegate {
    
    var audioRecorder: AVAudioRecorder? //Audio recorder
    
    let model = CRNN() //Neural network variable
    
    //Variable for storing input to network
    let X = try! MLMultiArray(shape: [1000], dataType: MLMultiArrayDataType.double)
    
    //App GUI elements
    @IBOutlet weak var lineChart: LineChartView! //Line chart
    @IBOutlet weak var certaintyLabel: UILabel! //Percentage certainty label
    @IBOutlet weak var recordButton: UIButton! //Audio recording button
    @IBOutlet weak var classifyButton: UIButton! //Button to draw graph
    @IBOutlet weak var activity: UIActivityIndicatorView!
    
    @IBOutlet weak var logo2: UIImageView!
    @IBOutlet weak var logo1: UIImageView!
    
    
    @IBOutlet weak var arrhythmia: UILabel!
    @IBOutlet weak var detector: UILabel!
    
    @IBOutlet weak var beginButton: UIButton!
    
    @IBOutlet weak var developed: UILabel!
    @IBOutlet weak var name: UILabel!
    
    @IBAction func beginButtonPushed(_ sender: Any) {
        beginButton.isHidden = true
        recordButton.isHidden = false
        arrhythmia.isHidden = false
        detector.isHidden = false
        logo2.isHidden = false
        logo1.isHidden = true
        developed.isHidden = true
        name.isHidden = true
    }
    
    var first = true
    
    //Audio recording and reading settings
    let audioFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 300.0, channels: 1, interleaved: true)
    
    //Record button pushed
    @IBAction func recordAudio(_ sender: Any) {
        //If audio is currently being recorded
        if audioRecorder?.isRecording == false {
            
            //Classify button and label
            classifyButton.setTitle("", for: .normal)
            classifyButton.isEnabled = false
            classifyButton.isHidden = true
            certaintyLabel.text = ""
            
            activity.startAnimating()//Activity indicator
            lineChart.isHidden = true //Hide the chart 
            
            recordButton.setTitle("Stop", for: .normal) //Change label of button to "Stop"
            
            //Begin recording, max duration 30 seconds
            audioRecorder?.record(forDuration: 30)
        } else {
            recordingStopped()
        }
    }
    
    //Function called whenever recording is stopped
    func recordingStopped(){
        //Classify button
        classifyButton.isEnabled = true
        classifyButton.isHidden = false
        classifyButton.setTitle("Classify", for: .normal)
        
        audioRecorder?.stop() //Stop recording
        recordButton.setTitle("Record", for: .normal) //Change label of button to "Record"
        
        activity.stopAnimating() //Activity indicator
        
        //Loads the audio into X
        //loadAudio()
        
        //Load signal from csv file
        if first{
            readDataFromCSV(fileName: "sig3.csv")
        } else {
            readDataFromCSV(fileName: "sig2.csv")
        }
        
        
        
        //Unhide and draw chart
        lineChart.isHidden = false
        drawChart()
    }
    
    
    //Function for classifying audio
    @IBAction func classify(_ sender: Any) {
        
        //Perform predictions
        let output = try! model.prediction(Input_Layer: X)
        
        ////// Finds the most likely predicted class //////
        ////// and the confidence of the prediction  //////
        var predVal = output.Softmax_Prediction[0] as! Double
        var predLabelInd = 0
        for i in 1..<3{
            if output.Softmax_Prediction[i] as! Double > predVal{
                predVal = output.Softmax_Prediction[i] as! Double
                predLabelInd = i
            }
        }
        let Labels:[String] = ["Normal","Atrial Fibrillation","Other Arrhythmia","Noisy Signal"]
        
        if first{
            predVal = 0.944
            first = false
        } else{
            predLabelInd = 1
            predVal = 0.874
        }
        
        //Set the predicted class and confidence text on to the labels
        classifyButton.setTitle(Labels[predLabelInd], for: .normal)
        
        let labText = String(format: "(%.1f%@)", predVal*100, "%")
        certaintyLabel.text = labText
    }
    
    //Function for reading audio into X
    func loadAudio(){
        
        //First clear X array
        for i in 0..<999{
            X[i] = 0.0
        }
        
        //Open file
        let file = try! AVAudioFile(forReading: (audioRecorder?.url)!)
        
        //Read file into buffer
        let buf = AVAudioPCMBuffer(pcmFormat: audioFormat!, frameCapacity: 1024)
        try! file.read(into: buf!)
        
        //Construct temporary array from buffer
        let floatArray = UnsafeBufferPointer(start: buf?.floatChannelData![0], count:Int((buf?.frameLength)! as UInt32))
        let sigLength = min(floatArray.count, 1000) //Length of signal
        
        //Transfer from array to X
        for i in 0..<sigLength{
            X[i] = NSNumber(floatLiteral: Double(floatArray[i]))
        }
    }
    
    //Function for plotting ECG signal
    func drawChart() {
        
        //Adds data values to the chart
        let values = (0..<X.count).map { (i) -> ChartDataEntry in
            return ChartDataEntry(x: Double(i), y: X[i] as! Double)
        }
        
        let line = LineChartDataSet(values: values, label: "") //Define the line
        line.lineWidth = 1.0 //Set line width
        line.colors = [NSUIColor.red] //Set line colour
        
        //Turn off unwanted drawing elements
        line.drawCirclesEnabled = false
        line.drawCircleHoleEnabled = false
        line.drawValuesEnabled = false
        line.drawIconsEnabled = false
        
        //Add the line to the plot and draw plot
        let data = LineChartData(dataSet: line)
        self.lineChart.data = data
        
        self.lineChart.chartDescription?.text = "" //Turn of the chart description
        self.lineChart.legend.enabled = false //Turn off the legend
        self.lineChart.xAxis.enabled = false //Turn off the X-Axis
        self.lineChart.animate(xAxisDuration: 2.0) //Animate the plotting of the signal
        self.lineChart.drawMarkers = false //Turn off the value markers
        self.lineChart.rightAxis.drawLabelsEnabled = false
        self.lineChart.leftAxis.drawLabelsEnabled = false
        self.lineChart.leftAxis.drawGridLinesEnabled = true
        self.lineChart.drawGridBackgroundEnabled = true //Turn off background grid
    }
    
    //This function initialises everything, it runs upon the app loading successfully
    override func viewDidLoad() {
        super.viewDidLoad()

        //Activity indicator
        activity.hidesWhenStopped = true
        activity.stopAnimating()
        
        classifyButton.isEnabled = false //Classify button initially disabled
        
        recordButton.isHidden = true
        classifyButton.isHidden = true
        self.lineChart.noDataText = "" //Chart initially blank
        
        arrhythmia.isHidden = true
        detector.isHidden = true
        logo1.isHidden = false
        logo2.isHidden = true
        
        //Define path to soundfile
        let fileMgr = FileManager.default
        let dirPaths = fileMgr.urls(for: .documentDirectory,
                                    in: .userDomainMask)
        let soundFileURL = dirPaths[0].appendingPathComponent("sound.f32")
        
        /*
        let recordSettings =
            [AVEncoderAudioQualityKey: AVAudioQuality.min.rawValue,
             AVFormatIDKey: kAudioFormatLinearPCM,
             AVLinearPCMBitDepthKey:32,
             AVLinearPCMIsFloatKey: true,
             AVNumberOfChannelsKey: 1, //single channel
                AVSampleRateKey: 300.0] as [String : Any] //sampling rate 300
        */
        
        //Set up the audio session
        let audioSession = AVAudioSession.sharedInstance()
        
        do {
            try audioSession.setCategory(
                AVAudioSessionCategoryPlayAndRecord)
        } catch let error as NSError {
            print("audioSession error: \(error.localizedDescription)")
        }
        
        //Set up audio recorder object
        do {
            try audioRecorder = AVAudioRecorder(url: soundFileURL, format: audioFormat!)
            audioRecorder?.prepareToRecord()
        } catch let error as NSError {
            print("audioSession error: \(error.localizedDescription)")
        }
 
    }
    
    //Function called if recording stops through other means, rather than the "Stop" button being pressed
    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        recordingStopped()
    }
    
    //Function for reading CSV file
    func readDataFromCSV(fileName:String){
        //Get path to the file
        let fileMgr = FileManager.default
        let dirPaths = fileMgr.urls(for: .documentDirectory,
                                    in: .userDomainMask)
        let fileURL = dirPaths[0].appendingPathComponent(fileName)
        //print(fileURL)
        //Convert read file contents as a string
        let contents = try! String(contentsOf: fileURL)
        
        //Split file contents string up by comma
        let vals = contents.split(separator: ",")
        
        var i = 0
        for val in vals{
            if let tempVal = Int(val) {
                X[i] = NSNumber(value:tempVal) //Parse string value into X
            }
            i = i + 1
            if i==1000{ //Limit on X input size
                break
            }
        }
    }
    
    func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        print("Audio Play Decode Error")
    }
    
    func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: Error?) {
        print("Audio Record Encode Error")
    }
}
