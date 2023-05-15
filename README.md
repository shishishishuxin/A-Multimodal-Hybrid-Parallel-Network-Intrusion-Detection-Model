A Multimodal Hybrid Parallel Network Intrusion Detection Model

With the rapid growth of Internet data traffic, the means of malicious attack become more diversified. The single modal intrusion detection model cannot fully exploit the rich feature information in the massive network traffic data, resulting in unsatisfactory detection results. To address this issue, this paper proposes a multimodal hybrid parallel network intrusion detection model (MHPN). The proposed model extracts network traffic features from two modalities: the statistical information of network traffic and the original load of traffic, and constructs appropriate neural network models for each modal information. Firstly, a two-branch convolutional neural network is combined with Long Short-Term Memory (LSTM) network to extract the spatio-temporal feature information of network traffic from the original load mode of traffic, and a convolutional neural network is used to extract the feature information of traffic statistics. Then, the feature information extracted from the two modalities is fused and fed to the CosMargin classifier for network traffic classification. The experimental results on the ISCX-IDS 2012 and CIC-IDS-2017 datasets show that the MHPN model outperforms the single-modal models and achieves an average accuracy of 99.98$\%$. The model also demonstrates strong robustness and a positive sample recognition rate
