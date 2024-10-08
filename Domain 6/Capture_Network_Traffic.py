import pyshark
from scapy.all import wrpcap, rdpcap, Ether, IP, TCP, UDP
import pandas as pd
from collections import defaultdict

def capture_and_save_to_pcap(interface='eth0', packet_count=10, output_file='captured_packets.pcap'):
    # Capture packets
    capture = pyshark.LiveCapture(interface=interface)
    capture.sniff(packet_count=packet_count)

    # Convert PyShark packets to Scapy packets
    scapy_packets = []
    for packet in capture:
        try:
            # Reconstruct Ethernet layer
            eth = Ether(src=packet.eth.src, dst=packet.eth.dst)
            
            # Reconstruct IP layer
            if 'IP' in packet:
                ip = IP(src=packet.ip.src, dst=packet.ip.dst)
                
                # Reconstruct TCP/UDP layer
                if 'TCP' in packet:
                    tcp = TCP(sport=int(packet.tcp.srcport), dport=int(packet.tcp.dstport))
                    scapy_packet = eth/ip/tcp
                elif 'UDP' in packet:
                    udp = UDP(sport=int(packet.udp.srcport), dport=int(packet.udp.dstport))
                    scapy_packet = eth/ip/udp
                else:
                    scapy_packet = eth/ip
            else:
                scapy_packet = eth

            scapy_packets.append(scapy_packet)
        except AttributeError:
            print(f"Skipping a packet due to missing attributes: {packet}")

    # Write packets to pcap file
    wrpcap(output_file, scapy_packets)
    print(f"Captured {len(scapy_packets)} packets and saved to {output_file}")

def extract_features(pcap_file):
    packets = rdpcap(pcap_file)
    
    features = defaultdict(lambda: defaultdict(int))
    
    for packet in packets:
        if IP in packet and (TCP in packet or UDP in packet):
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            protocol = packet[IP].proto
            
            # Count packets and bytes for each flow
            features[(src_ip, dst_ip, protocol)]['packet_count'] += 1
            features[(src_ip, dst_ip, protocol)]['total_bytes'] += len(packet)
            
            # Track unique ports
            if TCP in packet:
                features[(src_ip, dst_ip, protocol)]['src_ports'].add(packet[TCP].sport)
                features[(src_ip, dst_ip, protocol)]['dst_ports'].add(packet[TCP].dport)
            elif UDP in packet:
                features[(src_ip, dst_ip, protocol)]['src_ports'].add(packet[UDP].sport)
                features[(src_ip, dst_ip, protocol)]['dst_ports'].add(packet[UDP].dport)
    
    # Convert to DataFrame
    df_data = []
    for (src_ip, dst_ip, protocol), flow_features in features.items():
        df_data.append({
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'protocol': protocol,
            'packet_count': flow_features['packet_count'],
            'total_bytes': flow_features['total_bytes'],
            'unique_src_ports': len(flow_features['src_ports']),
            'unique_dst_ports': len(flow_features['dst_ports'])
        })
    
    df = pd.DataFrame(df_data)
    return df

def main():
    pcap_file = 'captured_packets.pcap'
    
    # Capture packets and save to PCAP
    capture_and_save_to_pcap(interface='eth0', packet_count=10, output_file=pcap_file)
    
    # Extract features
    traffic_profile = extract_features(pcap_file)
    
    print("\nTraffic Profile:")
    print(traffic_profile)
    
    # Save traffic profile to CSV
    traffic_profile.to_csv('traffic_profile.csv', index=False)
    print("\nTraffic profile saved to 'traffic_profile.csv'")

if __name__ == "__main__":
    main()
