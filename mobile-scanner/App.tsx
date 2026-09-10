import { CameraView, useCameraPermissions } from 'expo-camera';
import { StatusBar } from 'expo-status-bar';
import { useRef, useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';

const API_URL = process.env.EXPO_PUBLIC_API_URL ?? 'http://127.0.0.1:8000';

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const camera = useRef<CameraView>(null);

  if (!permission) return <View style={styles.container}><ActivityIndicator color="#e9c46a" /></View>;
  if (!permission.granted) {
    return <View style={styles.container}>
      <Text style={styles.title}>Field Scan</Text>
      <Text style={styles.copy}>Camera access is needed to inspect product labels.</Text>
      <Pressable style={styles.button} onPress={requestPermission}><Text style={styles.buttonText}>Allow camera</Text></Pressable>
      <StatusBar style="light" />
    </View>;
  }

  async function scan() {
    if (!camera.current) return;
    setScanning(true);
    setResult(null);
    try {
      const photo = await camera.current.takePictureAsync({ quality: 0.8 });
      if (!photo?.uri) throw new Error('Camera did not return an image');
      const form = new FormData();
      form.append('image', { uri: photo.uri, name: 'inspection.jpg', type: 'image/jpeg' } as any);
      const response = await fetch(`${API_URL}/scan`, { method: 'POST', body: form });
      if (!response.ok) throw new Error(`Scanner error ${response.status}`);
      setResult(await response.json());
    } catch (error) {
      setResult({ status: 'ERROR', message: error instanceof Error ? error.message : 'Scan failed' });
    } finally {
      setScanning(false);
    }
  }

  const statusColor = result?.status === 'FRESH' ? '#70c1b3' : result?.status === 'EXPIRED' ? '#e76f51' : '#e9c46a';
  return (
    <View style={styles.screen}>
      <View style={styles.header}><Text style={styles.eyebrow}>FOOD INSPECTOR</Text><Text style={styles.title}>Field Scan</Text></View>
      <View style={styles.cameraFrame}><CameraView ref={camera} style={StyleSheet.absoluteFill} facing="back" /></View>
      <Text style={styles.hint}>Frame the barcode and expiry print in one image</Text>
      <Pressable style={[styles.button, scanning && styles.disabled]} onPress={scan} disabled={scanning}>
        {scanning ? <ActivityIndicator color="#17212b" /> : <Text style={styles.buttonText}>Scan product</Text>}
      </Pressable>
      {result && <View style={styles.result}>
        <Text style={[styles.status, { color: statusColor }]}>{result.status}</Text>
        {result.expiry_date && <Text style={styles.detail}>Expiry: {result.expiry_date}</Text>}
        {result.barcodes?.map((barcode: any) => <Text style={styles.detail} key={barcode.value}>{barcode.type}: {barcode.value}</Text>)}
        {result.message && <Text style={styles.detail}>{result.message}</Text>}
      </View>}
      <StatusBar style="light" />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#17212b', padding: 24, paddingTop: 72 },
  container: {
    flex: 1,
    backgroundColor: '#17212b',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 28,
  },
  header: { marginBottom: 28 },
  eyebrow: { color: '#e9c46a', fontSize: 12, letterSpacing: 2, fontWeight: '700' },
  title: { color: '#f4f1de', fontSize: 34, fontWeight: '700', marginTop: 6 },
  copy: { color: '#bdc7cf', fontSize: 16, textAlign: 'center', marginVertical: 18 },
  cameraFrame: { height: 390, borderRadius: 18, overflow: 'hidden', borderWidth: 2, borderColor: '#e9c46a', backgroundColor: '#263644' },
  hint: { color: '#bdc7cf', textAlign: 'center', marginVertical: 18 },
  button: { backgroundColor: '#e9c46a', minHeight: 54, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  disabled: { opacity: 0.65 },
  buttonText: { color: '#17212b', fontSize: 17, fontWeight: '700' },
  result: { marginTop: 24, padding: 18, borderLeftWidth: 4, borderLeftColor: '#e9c46a', backgroundColor: '#263644' },
  status: { fontSize: 24, fontWeight: '800' },
  detail: { color: '#f4f1de', marginTop: 7, fontSize: 15 },
});
