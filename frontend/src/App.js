import './index.css'
import React, { useState, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Download, TrendingUp, Calendar } from 'lucide-react';

const API_BASE_URL = '/api';

const stores = [
  { name: '제주애월점', model: 'HGBR' },
  { name: '부산광안리점', model: 'CatBoost' },
  { name: '수원타임빌라스지점', model: 'CatBoost' },
  { name: '연남점', model: 'CatBoost' }
];

const StoreTab = ({ storeName, modelName }) => {
  // 검증 패널 상태
  const [validationQuery, setValidationQuery] = useState({
    startDate: '2025-10-20',
    endDate: '2025-10-27',
    productName: '(전체)'
  });
  const [validationData, setValidationData] = useState(null);
  const [validationLoading, setValidationLoading] = useState(false);

  // 상품 목록 상태
  const [products, setProducts] = useState(['(전체)']); 
  const [productsLoading, setProductsLoading] = useState(false); // 로딩 상태 추가

  // 예측 패널 상태
  const [forecastBaseDate, setForecastBaseDate] = useState('2025-10-03');
  const [forecastData, setForecastData] = useState(null);
  const [forecastLoading, setForecastLoading] = useState(false);


  // 상품 목록 가져오는 함수
  const fetchProducts = useCallback(async () => {
    setProductsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/products/${storeName}`);
      const data = await response.json(); 
      
      if (!response.ok) {
        throw new Error('상품 목록을 불러오는데 실패했습니다.');
      }
      
      const productList = ['(전체)', ...data]; 
      console.log(`[${storeName}] 로드된 상품 목록 (총 ${productList.length}개):`, productList);

      setProducts(productList);
    } catch (error) {
      console.error(`[${storeName}] 상품 목록 로드 실패:`, error);
      setProducts(['(전체)']); 
    } finally {
      setProductsLoading(false);
    }
  }, [storeName]);

  // 컴포넌트가 마운트되거나 'storeName'이 변경될 때 상품 목록 로드
  React.useEffect(() => {
    fetchProducts();
  }, [storeName, fetchProducts]); // storeName이 변경될 때마다 재실행

  // 검증 조회 API 호출
  const handleValidationQuery = async () => {
    setValidationLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          store_name: storeName,
          start_date: validationQuery.startDate,
          end_date: validationQuery.endDate,
          product_name: validationQuery.productName
        })
      });
      const data = await response.json();
      
      if (!response.ok) {
        // HTTP 오류 코드를 받았을 때
        const errorMessage = data.detail || '검증 조회 중 알 수 없는 오류 발생';
        throw new Error(errorMessage);
      }
      
      setValidationData(data);
    } catch (error) {
      console.error('검증 조회 실패:', error);
      alert('검증 데이터를 불러오는데 실패했습니다: ' + error.message);
    } finally {
      setValidationLoading(false);
    }
  };

  // 예측 생성 API 호출
  const handleForecastGenerate = async () => {
    setForecastLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/forecast`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          store_name: storeName,
          base_date: forecastBaseDate,
          horizon: 7
        })
      });
      const data = await response.json();
      setForecastData(data);
    } catch (error) {
      console.error('예측 생성 실패:', error);
      alert('예측을 생성하는데 실패했습니다.');
    } finally {
      setForecastLoading(false);
    }
  };

  // CSV 다운로드
  const handleDownloadCSV = () => {
    if (forecastData?.csv_filename) {
      window.open(`${API_BASE_URL}/download/${forecastData.csv_filename}`, '_blank');
    }
  };

  return (
    <div className="p-4">
      <div className="mb-4 text-sm text-gray-600">
        <strong>지점:</strong> {storeName} · 모델: {modelName}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1.45fr_1fr] gap-6">
        {/* 왼쪽: 검증 패널 */}
        <div className="bg-white rounded-lg border border-gray-200 p-5">
          <h2 className="text-xl font-bold mb-4">조회기간 · 검증</h2>

          {/* 검색 필터 */}
          <div className="grid grid-cols-4 gap-3 mb-4">
            <div>
              <label className="block text-sm font-medium mb-1">조회 시작일</label>
              <input
                type="date"
                value={validationQuery.startDate}
                onChange={(e) => setValidationQuery({...validationQuery, startDate: e.target.value})}
                className="w-full px-3 py-2 border rounded-lg"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">조회 종료일</label>
              <input
                type="date"
                value={validationQuery.endDate}
                onChange={(e) => setValidationQuery({...validationQuery, endDate: e.target.value})}
                className="w-full px-3 py-2 border rounded-lg"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">상품명</label>
              <select
                value={validationQuery.productName}
                onChange={(e) => setValidationQuery({...validationQuery, productName: e.target.value})}
                className="w-full px-3 py-2 border rounded-lg"
                disabled={productsLoading} // 상품 로딩 중에는 비활성화
              >
                {/* 로딩 중일 때 '로딩 중...' 표시 */}
                {productsLoading ? (
                  <option>로딩 중...</option>
                ) : (
                  // 상태로 관리되는 products를 매핑하여 옵션 생성
                  products.map(p => <option key={p} value={p}>{p}</option>)
                )}
              </select>
            </div>
            <div className="flex items-end">
              <button
                onClick={handleValidationQuery}
                disabled={validationLoading || productsLoading}
                className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
              >
                {validationLoading ? '조회중...' : '조회'}
              </button>
            </div>
          </div>

          {/* 검증 결과 */}
          {validationData && (
            <>
              {/* 차트 */}
              <div className="mb-4">
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={validationData.daily_chart_data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="actual_sales" stroke="#2563eb" name="실제판매량" strokeWidth={2} />
                    <Line type="monotone" dataKey="model_prediction" stroke="#16a34a" name="e시크 수요예측" strokeWidth={2} />
                    {validationData.daily_chart_data[0]?.avg_prediction && (
                      <Line type="monotone" dataKey="avg_prediction" stroke="#808080" name="과거수요예측" strokeWidth={2} />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* 테이블 */}
              <div className="overflow-auto max-h-64 border rounded-lg">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 sticky top-0">
                    <tr>
                      <th className="px-4 py-2 text-left">날짜(yy-mm-dd)</th>
                      <th className="px-4 py-2 text-right">실제판매량</th>
                      <th className="px-4 py-2 text-right">e시크 수요예측</th>
                      <th className="px-4 py-2 text-right">오차</th>
                    </tr>
                  </thead>
                  <tbody>
                    {validationData.daily_table_data.map((row, idx) => (
                      <tr key={idx} className="border-t hover:bg-gray-50">
                        <td className="px-4 py-2">{row.date}</td>
                        <td className="px-4 py-2 text-right">{row.actual_sales}</td>
                        <td className="px-4 py-2 text-right">{row.model_prediction.toFixed(1)}</td>
                        <td className="px-4 py-2 text-right">{row.error.toFixed(0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          {!validationData && (
            <div className="text-center py-12 text-gray-400">
              조회 버튼을 눌러 검증 데이터를 확인하세요
            </div>
          )}
        </div>

        {/* 오른쪽: 예측 패널 */}
        <div className="bg-white rounded-lg border border-gray-200 p-5">
          <h2 className="text-xl font-bold mb-4">기준일자 · 7일 예측</h2>

          {/* 기준일자 입력 */}
          <div className="grid grid-cols-[1fr_auto] gap-3 mb-4">
            <div>
              <label className="block text-sm font-medium mb-1">기준일자</label>
              <input
                type="date"
                value={forecastBaseDate}
                onChange={(e) => setForecastBaseDate(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg"
              />
            </div>
            <div className="flex items-end">
              <button
                onClick={handleForecastGenerate}
                disabled={forecastLoading}
                className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:opacity-50 whitespace-nowrap"
              >
                {forecastLoading ? '생성중...' : '다음주 예측 생성하기'}
              </button>
            </div>
          </div>

          {/* 예측 결과 */}
          {forecastData && (
            <>
              <div className="mb-3 text-sm font-medium text-green-700">
                예측 완료! 아래 버튼으로 CSV 저장하세요.
              </div>

              <button
                onClick={handleDownloadCSV}
                className="w-full mb-4 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center justify-center gap-2"
              >
                <Download size={18} />
                다음주 예측 CSV 다운로드
              </button>

              {/* 예측 테이블 */}
              <div className="overflow-auto max-h-96 border rounded-lg">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 sticky top-0">
                    <tr>
                      <th className="px-3 py-2 text-left">날짜</th>
                      <th className="px-3 py-2 text-left">상품명</th>
                      <th className="px-3 py-2 text-right">예측수량</th>
                      <th className="px-3 py-2 text-right">주문량_ceil</th>
                    </tr>
                  </thead>
                  <tbody>
                    {forecastData.predictions.map((row, idx) => (
                      <tr key={idx} className="border-t hover:bg-gray-50">
                        <td className="px-3 py-2">{row.date}</td>
                        <td className="px-3 py-2">{row.product_name}</td>
                        <td className="px-3 py-2 text-right">{row.predicted_qty.toFixed(1)}</td>
                        <td className="px-3 py-2 text-right">{row.order_qty_ceil}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          {!forecastData && (
            <div className="text-center py-12 text-gray-400">
              예측 생성 버튼을 눌러 다음주 예측을 확인하세요
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const App = () => {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 헤더 */}
      <div className="bg-amber-50 border-b border-amber-100 px-6 py-5">
        <h1 className="text-3xl font-bold text-gray-800 flex items-center gap-2">
          🍩 Randy's Donuts · 머신러닝 기반 수요예측 시스템
        </h1>
      </div>

      {/* 탭 네비게이션 */}
      <div className="bg-white border-b border-gray-200">
        <div className="flex gap-1 px-4">
          {stores.map((store, idx) => (
            <button
              key={store.name}
              onClick={() => setActiveTab(idx)}
              className={`px-6 py-3 font-medium border-b-2 transition-colors ${
                activeTab === idx
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-800'
              }`}
            >
              {store.name}
            </button>
          ))}
        </div>
      </div>

      {/* 탭 콘텐츠 */}
      <div className="max-w-7xl mx-auto">
        {stores.map((store, idx) => (
          <div key={store.name} className={activeTab === idx ? 'block' : 'hidden'}>
            <StoreTab storeName={store.name} modelName={store.model} />
          </div>
        ))}
      </div>
    </div>
  );
};

export default App;
